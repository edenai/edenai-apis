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
            confidence=original_response['attributes'].get('face', {}).get('confidence'),
            landmarks=FaceLandmarks(
                left_eye=[original_response.get('eye_left', {}).get('x'), original_response.get('eye_left', {}).get('y')],
                right_eye=[original_response.get('eye_right', {}).get('x'), original_response.get('eye_right', {}).get('y')],
                mouth_center=[original_response.get('mouse_center', {}).get('x'), original_response.get('mouse_center', {}).get('y')],
                nose_tip=[original_response.get('eye_left', {}).get('x'), original_response.get('eye_left', {}).get('y')],
            ),
            emotions=FaceEmotions(
                joy=original_response['attributes'].get('happiness', {}).get('confidence'),
                sorrow=original_response['attributes'].get('sadness', {}).get('confidence'),
                anger=original_response['attributes'].get('anger', {}).get('confidence'),
                surprise=original_response['attributes'].get('surprise', {}).get('confidence'),
                fear=original_response['attributes'].get('fear', {}).get('confidence'),
                disgust=original_response['attributes'].get('disgust', {}).get('confidence'),
                neutral=original_response['attributes'].get('neutral_mood', {}).get('confidence'),
            ),
            age=original_response['attributes'].get('age_est', {}).get('value'),
            gender=original_response['attributes'].get('gender', {}).get('value'),
            facial_hair=FaceFacialHair(
                moustache=1.0 if original_response['attributes'].get('mustache', {}).get('value') == 'true' else 0.0,
                beard=1.0 if original_response['attributes'].get('beard', {}).get('value') == 'true' else 0.0
            ),
            accessories=FaceAccessories(
                sunglasses=1.0 if original_response['attributes'].get('dark_glasses', {}).get('value') == 'true' else 0.0,
                eyeglasses=1.0 if original_response['attributes'].get('glasses', {}).get('value') == 'true' else 0.0,
                headwear=1.0 if original_response['attributes'].get('hat', {}).get('value') == 'true' else 0.0
            ),
            features=FaceFeatures(
                eyes_open=1.0 if original_response['attributes'].get('eyes', {}).get('value') == 'open' else 0.0,
                smile=1.0 if original_response['attributes'].get('smiling', {}).get('value') == 'true' else 0.0,
                mouth_open=1.0 if original_response['attributes'].get('lips', {}).get('value', 'sealed') != 'sealed' else 0.0,
            ),
        ))

        standardized_response = FaceDetectionDataClass(items=items)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
