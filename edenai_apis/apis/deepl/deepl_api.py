from io import BufferedReader
from time import sleep
import requests
import base64
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.translation.automatic_translation import AutomaticTranslationDataClass
from edenai_apis.features.translation.document_translation.document_translation_dataclass import DocumentTranslationDataClass
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType


class DeeplApi(ProviderInterface, TranslationInterface):
    provider_name = "deepl"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.url = self.api_settings["url"]
        self.header = {
            "authorization": f"DeepL-Auth-Key {self.api_key}",
        }

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        url = f"{self.url}translate"

        data = {
            "text": text,
            "source_lang": source_language,
            "target_lang": target_language,
        }

        response = requests.request("POST", url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)


        standardized_response = AutomaticTranslationDataClass(
            text=original_response['translations'][0]['text']
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=original_response, standardized_response=standardized_response
        )

    def translation__document_translation(
        self,
        file: BufferedReader,
        source_language: str,
        target_language: str
    ) -> ResponseType[DocumentTranslationDataClass]:
        files = {
            'file': file,
        }

        data = {
            'target_lang': target_language,
            'source_lang': source_language
        }

        response = requests.post(f'{self.url}document', headers=self.header, data=data, files=files)
        original_response = response.json()
        
        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)
        
        document_id, document_key = response.json().values()

        doc_key = {
            'document_key': document_key
        }

        response = requests.post(f'{self.url}document/{document_id}', headers=self.header, data=doc_key).json()
        while response['status'] != 'done':
            response = requests.post(f'{self.url}document/{document_id}', headers=self.header, data=doc_key).json()
            if response['status'] == 'error':
                raise ProviderException(response['error_message'])
            sleep(.5)

        response = requests.post(f'{self.url}document/{document_id}/result', headers=self.header, data=doc_key)

        b64_file = base64.b64encode(response.content)
        std_resp = DocumentTranslationDataClass(file=b64_file)
        return ResponseType[DocumentTranslationDataClass](
            original_response=response.content, standardized_response=std_resp
        )
