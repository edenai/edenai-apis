import base64
import mimetypes
from io import BytesIO
from typing import Sequence, Optional

from google.protobuf.json_format import MessageToDict

from edenai_apis.apis.google.google_helpers import handle_google_call
from edenai_apis.features.translation.automatic_translation import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.document_translation import (
    DocumentTranslationDataClass,
)
from edenai_apis.features.translation.language_detection import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import upload_file_bytes_to_s3, USER_PROCESS


class GoogleTranslationApi(TranslationInterface):
    def translation__automatic_translation(
        self,
        source_language: str,
        target_language: str,
        text: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[AutomaticTranslationDataClass]:
        # Getting response
        client = self.clients["translate"]
        parent = f"projects/{self.project_id}/locations/global"

        payload = {
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": source_language,
            "target_language_code": target_language,
        }
        response = handle_google_call(client.translate_text, **payload)

        # Analyze response
        # Getting the translated text
        data = response.translations
        res = data[0].translated_text
        std: AutomaticTranslationDataClass
        if res != "":
            std = AutomaticTranslationDataClass(text=res)
        else:
            raise ProviderException("Empty Text was returned")
        return ResponseType[AutomaticTranslationDataClass](
            original_response=MessageToDict(response._pb), standardized_response=std
        )

    def translation__language_detection(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[LanguageDetectionDataClass]:

        payload = {
            "parent": f"projects/{self.project_id}/locations/global",
            "content": text,
            "mime_type": "text/plain",
        }
        response = handle_google_call(
            self.clients["translate"].detect_language, **payload
        )

        items: Sequence[InfosLanguageDetectionDataClass] = []
        for language in response.languages:
            items.append(
                InfosLanguageDetectionDataClass(
                    language=language.language_code,
                    display_name=get_language_name_from_code(
                        isocode=language.language_code
                    ),
                    confidence=language.confidence,
                )
            )
        return ResponseType[LanguageDetectionDataClass](
            original_response=MessageToDict(response._pb),
            standardized_response=LanguageDetectionDataClass(items=items),
        )

    def translation__document_translation(
        self,
        file: str,
        source_language: str,
        target_language: str,
        file_type: str,
        file_url: str = "",
        **kwargs,
    ) -> ResponseType[DocumentTranslationDataClass]:
        mimetype = mimetypes.guess_type(file)[0]
        extension = mimetypes.guess_extension(mimetype)
        client = self.clients["translate"]
        parent = f"projects/{self.project_id}/locations/global"

        with open(file, "rb") as file_:
            document_input_config = {
                "content": file_.read(),
                "mime_type": file_type,
            }

            payload = {
                "request": {
                    "parent": parent,
                    "target_language_code": target_language,
                    "source_language_code": source_language,
                    "document_input_config": document_input_config,
                }
            }
            original_response = handle_google_call(client.translate_document, **payload)

            file_bytes = original_response.document_translation.byte_stream_outputs[0]

        serialized_response = MessageToDict(original_response._pb)

        b64_file = base64.b64encode(file_bytes)
        resource_url = upload_file_bytes_to_s3(
            BytesIO(file_bytes), extension, USER_PROCESS
        )

        return ResponseType[DocumentTranslationDataClass](
            original_response=serialized_response,
            standardized_response=DocumentTranslationDataClass(
                file=b64_file, document_resource_url=resource_url
            ),
        )
