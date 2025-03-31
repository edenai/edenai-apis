from typing import Dict, List, Optional

from edenai_apis.features import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import (
    ProviderException,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
)


class VoxistApi(ProviderInterface, AudioInterface):
    provider_name: str = "voxist"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings: Dict = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        raise ProviderException(
            message="This provider is deprecated.",
            code=500,
        )

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        raise ProviderException(
            message="This provider is deprecated.",
            code=500,
        )
