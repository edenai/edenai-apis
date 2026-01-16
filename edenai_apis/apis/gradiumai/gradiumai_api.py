import asyncio
import base64
import json
from io import BytesIO
from typing import Dict, Optional

import websockets

from edenai_apis.features.audio import AudioInterface
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    aupload_file_bytes_to_s3,
)

from .config import (
    voice_ids,
    DEFAULT_FORMAT,
    REGIONS,
    DEFAULT_REGION,
    DEFAULT_VOICE_NAME,
)

# Create lowercase lookup for case-insensitive voice matching
_voice_ids_lower = {k.lower(): v for k, v in voice_ids.items()}


class GradiumaiApi(ProviderInterface, AudioInterface):
    provider_name = "gradiumai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings.get("api_key", "")
        self.region = self.api_settings.get("region", DEFAULT_REGION)
        self.ws_url = f"wss://{REGIONS.get(self.region, REGIONS[DEFAULT_REGION])}/api/speech/tts"

    async def audio__atts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: Optional[float] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TextToSpeechDataClass]:
        """Convert text to speech using Gradium AI WebSocket API.

        Args:
            text: The text to convert to speech
            model: The model name (not used, for interface compatibility)
            voice: The voice ID or name (e.g., "Emma", "YTpq7expH9539ERJ").
                   Defaults to "Emma"
            audio_format: Audio format (pcm, wav, opus). Defaults to "pcm"
                         Note: Gradium streams PCM by default
            speed: Not directly supported (ignored)
            provider_params: Provider-specific settings
        """
        provider_params = provider_params or {}

        resolved_voice = voice or DEFAULT_VOICE_NAME

        # Resolve voice name to voice ID (case-insensitive lookup using lowercase)
        voice_lower = resolved_voice.lower()
        if voice_lower in _voice_ids_lower:
            voice_id = _voice_ids_lower[voice_lower]
        else:
            # Assume it's already a voice ID
            voice_id = resolved_voice

        # Gradium supports pcm, wav, opus for streaming
        resolved_format = audio_format if audio_format in ["pcm", "wav", "opus"] else "wav"

        headers = {
            "x-api-key": self.api_key,
            "x-api-source": "edenai",
        }

        audio_chunks = []

        try:
            async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
                # Send setup message
                setup_msg = {
                    "type": "setup",
                    "voice_id": voice_id,
                    "output_format": resolved_format,
                }
                await ws.send(json.dumps(setup_msg))

                # Wait for ready message
                ready_response = await ws.recv()
                ready_data = json.loads(ready_response)
                if ready_data.get("type") == "error":
                    raise ProviderException(
                        ready_data.get("message", "Setup failed"),
                        code=400
                    )

                # Send text message
                text_msg = {
                    "type": "text",
                    "text": text,
                }
                await ws.send(json.dumps(text_msg))

                # Send end of stream
                await ws.send(json.dumps({"type": "end_of_stream"}))

                # Collect audio chunks
                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        break
                    except websockets.exceptions.ConnectionClosed:
                        break

                    # Check if response is binary (raw audio) or text (JSON)
                    if isinstance(response, bytes):
                        audio_chunks.append(response)
                        continue

                    data = json.loads(response)
                    msg_type = data.get("type")

                    if msg_type == "audio":
                        # Audio data is base64 encoded in the "audio" field
                        audio_b64 = data.get("audio", "")
                        if audio_b64:
                            audio_chunks.append(base64.b64decode(audio_b64))
                    elif msg_type == "end_of_stream":
                        break
                    elif msg_type == "error":
                        raise ProviderException(
                            data.get("message", "TTS generation failed"),
                            code=400
                        )
                    # Skip other message types (ready, text, etc.)

            # Combine all audio chunks
            combined_audio = b"".join(audio_chunks)
            audio_content = BytesIO(combined_audio)
            audio = base64.b64encode(combined_audio).decode("utf-8")

            audio_content.seek(0)
            resource_url = await aupload_file_bytes_to_s3(
                audio_content, f".{resolved_format}", USER_PROCESS
            )

            return ResponseType[TextToSpeechDataClass](
                original_response={},
                standardized_response=TextToSpeechDataClass(
                    audio=audio, voice_type=1, audio_resource_url=resource_url
                ),
            )
        except websockets.exceptions.WebSocketException as exc:
            raise ProviderException(str(exc), code=500)
