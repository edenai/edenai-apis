import base64
from io import BytesIO
from typing import Dict, Optional

import httpx

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
from edenai_apis.utils.http_client import async_client, AUDIO_TIMEOUT

from .config import (
    voice_ids,
    DEFAULT_FORMAT,
    REGIONS,
    DEFAULT_REGION,
    DEFAULT_VOICE_NAME,
)


class GradiumaiApi(ProviderInterface, AudioInterface):
    provider_name = "gradiumai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings.get("api_key", "")
        self.region = self.api_settings.get("region", DEFAULT_REGION)
        self.base_url = f"https://{REGIONS.get(self.region, REGIONS[DEFAULT_REGION])}"

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
        """Convert text to speech using Gradium AI API.

        Args:
            text: The text to convert to speech
            model: The model name (default: "default")
            voice: The voice ID or name (e.g., "Emma", "YTpq7expH9539ERJ").
                   Defaults to "Emma"
            audio_format: Audio format (mp3, wav, pcm, opus). Defaults to "mp3"
            speed: Not directly supported (ignored)
            provider_params: Provider-specific settings:
                - region: API region ("us" or "eu", default "us")
                - padding_bonus: Padding bonus value
                - temp: Temperature for generation
                - cfg_coef: CFG coefficient
        """
        provider_params = provider_params or {}

        # Set defaults
        resolved_model = model or "default"
        resolved_voice = voice or DEFAULT_VOICE_NAME

        # Resolve voice name to voice ID if it's a name
        if resolved_voice in voice_ids:
            voice_id = voice_ids[resolved_voice]
        else:
            # Assume it's already a voice ID
            voice_id = resolved_voice

        # Resolve audio format
        resolved_format = audio_format or DEFAULT_FORMAT

        # Build request payload
        url = f"{self.base_url}/api/speech/tts"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build setup configuration
        setup_config = {
            "model_name": resolved_model,
            "voice_id": voice_id,
            "output_format": resolved_format,
        }

        # Add optional json_config parameters
        json_config = {}
        if provider_params.get("padding_bonus") is not None:
            json_config["padding_bonus"] = provider_params["padding_bonus"]
        if provider_params.get("temp") is not None:
            json_config["temp"] = provider_params["temp"]
        if provider_params.get("cfg_coef") is not None:
            json_config["cfg_coef"] = provider_params["cfg_coef"]

        if json_config:
            setup_config["json_config"] = json_config

        payload = {
            "setup": setup_config,
            "text": text,
        }

        try:
            async with async_client(AUDIO_TIMEOUT) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                # Get audio content from response
                audio_content = BytesIO(response.content)
                audio = base64.b64encode(audio_content.read()).decode("utf-8")

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
        except httpx.HTTPStatusError as exc:
            raise ProviderException(exc.response.text, code=exc.response.status_code)
