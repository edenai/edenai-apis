from io import BufferedReader
from typing import List

import requests
from edenai_apis.features.image.face_detection import FaceDetectionDataClass
from edenai_apis.features.image.face_detection.face_detection_dataclass import FaceItem
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from features import ProviderInterface
from features.image import ImageInterface

class SkybiometryApi(ProviderInterface, ImageInterface):
    provider_name = 'skybiometry'

    def __init__(self) -> None:
        self.base_url = 'https://api.skybiometry.com/fc/'
        self.settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.settings['api_key']
        self.api_secret = self.settings['api_secret']

    def image__face_detection(self, file: BufferedReader) -> ResponseType[FaceDetectionDataClass]:
        files = {
            'file': file
        }

        response = requests.post(f'{self.base_url}faces/detect.json?api_key={self.api_key}&api_secret={self.api_secret}&attributes=all', files=files)

        if response.status_code != 200:
            raise ProviderException(
                message=response.json()['error_message'],
                code=response.status_code
            )

        original_response = response.json()
        items: List[FaceItem] = []

        standardized_response = FaceDetectionDataClass(items=items)
        return response