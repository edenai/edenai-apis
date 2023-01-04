from io import BufferedReader
from pprint import pprint
from typing import List

import requests
from edenai_apis.features import ProviderInterface
from edenai_apis.features.image import ImageInterface
from edenai_apis.features.image import FaceItem, FaceDetectionDataClass
from edenai_apis.features.image.face_detection.face_detection_dataclass import FaceAccessories, FaceEmotions, FaceFacialHair, FaceFeatures, FaceLandmarks
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum

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

        endpoint = f'{self.base_url}faces/detect.json'
        query_params = f'api_key={self.api_key}&api_secret={self.api_secret}&attributes=all'
        response = requests.post(f'{endpoint}?{query_params}', files=files)

        original_response = response.json()
        if response.status_code != 200:
            raise ProviderException(
                message=original_response['error_message'],
                code=response.status_code
            )


        original_response = original_response['photos'][0]['tags'][0]
        pprint(original_response)
        items: List[FaceItem] = []
        items.append(FaceItem(
            confidence=original_response['attributes']['face']['confidence'],
            landmarks=FaceLandmarks(
                left_eye=[original_response['eye_left']['x'], original_response['eye_left']['y']],
                right_eye=[original_response['eye_right']['x'], original_response['eye_right']['y']],
                mouth_center=[original_response['mouth_center']['x'], original_response['mouth_center']['y']],
                nose_tip=[original_response['nose']['x'], original_response['nose']['y']],
            ),
            emotions=FaceEmotions(
                joy=original_response['attributes']['happiness']['confidence'],
                sorrow=original_response['attributes']['sadness']['confidence'],
                anger=original_response['attributes']['anger']['confidence'],
                surprise=original_response['attributes']['surprise']['confidence'],
                fear=original_response['attributes']['fear']['confidence'],
                disgust=original_response['attributes']['disgust']['confidence'],
                neutral=original_response['attributes']['neutral_mood']['confidence'],
            ),
            age=original_response['attributes']['age_est']['value'],
            gender=original_response['attributes']['gender']['value'],
            facial_hair=FaceFacialHair(
                moustache=1.0 if original_response['attributes']['mustache']['value'] == 'true' else 0.0,
                beard=1.0 if original_response['attributes']['beard']['value'] == 'true' else 0.0
            ),
            accessories=FaceAccessories(
                sunglasses=1.0 if original_response['attributes']['dark_glasses']['value'] == 'true' else 0.0,
                eyeglasses=1.0 if original_response['attributes']['glasses']['value'] == 'true' else 0.0,
                headwear=1.0 if original_response['attributes']['hat']['value'] == 'true' else 0.0
            ),
            features=FaceFeatures(
                eyes_open=1.0 if original_response['attributes']['eyes']['value'] == 'open' else 0.0,
                smile=1.0 if original_response['attributes']['smiling']['value'] == 'true' else 0.0,
                mouth_open=1.0 if original_response['attributes']['lips']['value'] != 'sealed' else 0.0,
            ),
        ))

        standardized_response = FaceDetectionDataClass(items=items)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
