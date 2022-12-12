from asyncio import sleep
import datetime
from pprint import pprint
import sys
from collections import defaultdict
import base64
import json
from pathlib import Path
import time
from typing import Dict, List, Sequence
from io import BufferedReader, BytesIO
import requests
from PIL import Image as Img
from pdf2image.pdf2image import convert_from_bytes
import azure.cognitiveservices.speech as speechsdk
from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    Audio,
    TextToSpeechDataClass,
    SpeechDiarization,
    SpeechDiarizationEntry
)

from edenai_apis.features.base_provider.provider_api import ProviderApi
from edenai_apis.features.image import (
    ExplicitContentDataClass,
    ExplicitItem, LogoBoundingPoly,
    LogoDetectionDataClass,
    LogoVertice,
    FaceDetectionDataClass,
    LogoItem,
    LandmarkDetectionDataClass, LandmarkItem,
    ObjectDetectionDataClass, ObjectItem
)
from edenai_apis.features.image.image_class import Image

from edenai_apis.features.ocr import (
    Bounding_box, OcrDataClass, Taxes,
    InfosReceiptParserDataClass, ItemLines,
    MerchantInformation, ReceiptParserDataClass, PaymentInformation,
    InvoiceParserDataClass,
    IdentityParserDataClass,
    InfosIdentityParserDataClass,
    get_info_country,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import InfoCountry, ItemIdentityParserDataClass, format_date
from edenai_apis.features.ocr.ocr_class import Ocr
from edenai_apis.features.text import (
    InfosKeywordExtractionDataClass, KeywordExtractionDataClass,
    InfosNamedEntityRecognitionDataClass, NamedEntityRecognitionDataClass,
    SentimentAnalysisDataClass, SummarizeDataClass
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import SegmentSentimentAnalysisDataClass
from edenai_apis.features.text.text_class import Text
from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass
)
from edenai_apis.features.translation.translation_class import Translation
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from edenai_apis.utils.audio import wav_converter
from edenai_apis.utils.conversion import format_string_url_language
from edenai_apis.utils.exception import ProviderException, LanguageException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncErrorResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType
)
from edenai_apis.utils.upload_s3 import upload_file_to_s3
from edenai_apis.apis.microsoft.microsoft_helpers import (
    content_processing,
    get_microsoft_headers,
    get_microsoft_urls,
    miscrosoft_normalize_face_detection_response,
    normalize_invoice_result,
)
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from .config import audio_voice_ids
from edenai_apis.utils.audio import wav_converter

class MicrosoftApi(
    ProviderApi,
    Image,
    Text,
    Translation,
    Ocr,
    Audio
):
    provider_name = "microsoft"

    def __init__(self, user=None):
        super().__init__()

        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.headers = get_microsoft_headers()
        self.url = get_microsoft_urls()
        self.user = user

    def ocr__ocr(self,
        file: BufferedReader,
        language: str
    ) -> ResponseType[OcrDataClass]:

        file_content = file.read()

        url = f"{self.api_settings['vision']['url']}/ocr?detectOrientation=true"

        response = requests.post(
            format_string_url_language(url, language, "language", self.provider_name)
            if language
            else url,
            headers=self.headers["vision"],
            data=file_content,
        ).json()

        final_text = ""

        if "error" in response:
            raise Exception(response["error"]["message"])

        # Get width and hight
        width, height = Img.open(file).size

        boxes: Sequence[Bounding_box] = []
        # Get region of text
        for region in response["regions"]:
            # Read line by region
            for line in region["lines"]:
                for word in line["words"]:
                    final_text += " " + word["text"]
                    boxes.append(
                        Bounding_box(
                            text=word["text"],
                            left=float(word["boundingBox"].split(",")[0]) / width,
                            top=float(word["boundingBox"].split(",")[1]) / height,
                            width=float(word["boundingBox"].split(",")[2]) / width,
                            height=float(word["boundingBox"].split(",")[3]) / height,
                        )
                    )
        standarized = OcrDataClass(
            text=final_text.replace("\n", " ").strip(), bounding_boxes=boxes
        )

        return ResponseType[OcrDataClass](
            original_response=response, standarized_response=standarized
        )


    def audio__text_to_speech(self,
        language: str,
        text: str,
        option: str
    ) -> ResponseType[TextToSpeechDataClass]:
        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_settings["speech"]["subscription_key"],
            region=self.api_settings["speech"]["service_region"],
        )

        if not language:
            raise ProviderException(
                f"language code: {language} badly formatted or "
                f"not supported by {self.provider_name}"
            )
        voiceid = audio_voice_ids[language][option]

        speech_config.speech_synthesis_voice_name = voiceid
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )

        # Getting response of API
        # output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        response = speech_synthesizer.speak_text_async(text).get()

        if response.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = response.cancellation_details

            raise ProviderException(
                "error",
                f"Speech synthesis canceled: {cancellation_details.reason}"
            )

        audio = base64.b64encode(response.audio_data).decode("utf-8")
        voice_type = 1

        standarized_response = TextToSpeechDataClass(audio=audio, voice_type=voice_type)

        return ResponseType[TextToSpeechDataClass](
            original_response={},
            standarized_response=standarized_response
        )


    def ocr__invoice_parser(self,
        file: BufferedReader,
        language: str
    ) -> ResponseType[InvoiceParserDataClass]:
        invoice_file_content = file

        # Get result
        document_analysis_client = FormRecognizerClient(
            endpoint=self.url["ocr_tables_async"],
            credential=AzureKeyCredential(
                self.api_settings["ocr_invoice"]["subscription_key"]
            ),
        )
        poller = document_analysis_client.begin_recognize_invoices(
            invoice=invoice_file_content
        )
        invoices = poller.result()

        result = [el.to_dict() for el in invoices]

        return ResponseType[InvoiceParserDataClass](
            original_response=result,
            standarized_response=normalize_invoice_result(result)
        )



    def ocr__receipt_parser(self,
        file: BufferedReader,
        language: str
    ) -> ResponseType[ReceiptParserDataClass]:

        receipt_file_content = file
        document_analysis_client = FormRecognizerClient(
            endpoint=self.url["ocr_tables_async"],
            credential=AzureKeyCredential(
                self.api_settings["ocr_invoice"]["subscription_key"]
            ),
        )
        poller = document_analysis_client.begin_recognize_receipts(receipt_file_content)
        form_pages = poller.result()
        result = [el.to_dict() for el in form_pages]
    
        # Normalize the response
        default_dict = defaultdict(lambda: None)
        fields = result[0].get("fields", default_dict)
        # 1. Invoice number
        invoice_total = fields.get("Total",default_dict).get("value")

        # 2. Date & time
        date = fields.get("TransactionDate", default_dict).get("value")
        time = fields.get("TransactionTime", default_dict).get("value")

        # 3. invoice_subtotal
        sub_total = fields.get("Subtotal", default_dict).get("value")

        # 4. merchant informations
        merchant = MerchantInformation(
            merchant_name=fields.get("MerchantName", default_dict).get("value"),
            merchant_address = fields.get("MerchantAddress", default_dict).get("value"),
            merchant_phone = fields.get("MerchantPhoneNumber", default_dict).get("vale")
        )

        # 5. Taxes
        taxes = [Taxes(taxes=fields.get("Tax", default_dict).get("value"))]

        # 6. Receipt infos / payment informations
        receipt_infos = fields.get("ReceiptType")
        payment_infos = PaymentInformation(
            tip = fields.get("Tip", default_dict).get("value")
        )

        # 7. Items
        items = []
        for item in fields.get("Items", default_dict).get("value", []):
            description = item["value"].get("Name", default_dict).get("value")
            price = item["value"].get("Price", default_dict).get("value")
            quantity = int(item["value"].get("Quantity", default_dict).get("value"))
            total = item["value"].get("TotalPrice", default_dict).get("value")
            items.append(
                ItemLines(
                    amount=total,
                    description=description,
                    unit_price=price,
                    quantity=quantity,
                )
            )

        receipt = InfosReceiptParserDataClass(
            item_lines=items,
            taxes=taxes,
            merchant_information=merchant,
            invoice_subtotal=sub_total,
            invoice_total=invoice_total,
            date=str(date),
            time = str(time),
            payment_information = payment_infos,
            receipt_infos=receipt_infos,
        )
        return ResponseType[ReceiptParserDataClass](
            original_response=result,
            standarized_response=ReceiptParserDataClass(extracted_data=[receipt])
        )

    def ocr__identity_parser(self, file: BufferedReader) -> ResponseType[IdentityParserDataClass]:
        file_content = file.read()

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_settings["ocr_id"]["subscription_key"],
            "Content-Type": "application/octet-stream",
        }

        response = requests.post(
            url=f"{self.api_settings['ocr_id']['url']}:analyze?{self.api_settings['ocr_id']['api_version']}",
            headers=headers,
            data=file_content
        )

        request_id = response.headers['apim-request-id']

        sleep(2000)

        response = requests.get(
            url=f"{self.api_settings['ocr_id']['url']}/analyzeResults/{request_id}?{self.api_settings['ocr_id']['api_version']}",
            headers=headers,
        )

        while response.json()['status'] == 'running':
            response = requests.get(
            url=f"{self.api_settings['ocr_id']['url']}/analyzeResults/{request_id}?{self.api_settings['ocr_id']['api_version']}",
            headers=headers,
            )
            sleep(500)

        original_response = response.json()
        microsoft_data = original_response['analyzeResult']

        items = []

        for document in microsoft_data['documents']:
            fields = document['fields']
            country = get_info_country(key=InfoCountry.ALPHA3, value=fields.get('CountryRegion', {}).get('content'))
            if country:
                country['confidence'] = fields.get('CountryRegion', {}).get('confidence')

            given_names=fields.get('FirstName', {}).get('content', "").split(' ')
            final_given_names = []
            for given_name in given_names:
                final_given_names.append(ItemIdentityParserDataClass(
                    value=given_name,
                    confidence=fields.get('FirstName', {}).get('confidence')
                ))

            items.append(InfosIdentityParserDataClass(
                document_type=ItemIdentityParserDataClass(
                    value=document.get('docType'),
                    confidence=document.get('confidence')
                ),
                country=country,
                birth_date=ItemIdentityParserDataClass(
                    value=fields.get('DateOfBirth', {}).get('valueDate'),
                    confidence=fields.get('DateOfBirth', {}).get('confidence')
                ),
                expire_date=ItemIdentityParserDataClass(
                    value=fields.get('DateOfExpiration', {}).get('valueDate'),
                    confidence=fields.get('DateOfExpiration', {}).get('confidence')
                ),
                issuance_date=ItemIdentityParserDataClass(
                    value=fields.get('DateOfIssue', {}).get('valueDate'),
                    confidence=fields.get('DateOfIssue', {}).get('confidence')
                ),
                issuing_state=ItemIdentityParserDataClass(
                    value=fields.get('IssuingAuthority', {}).get('content'),
                    confidence=fields.get('IssuingAuthority', {}).get('confidence'),
                ),
                document_id=ItemIdentityParserDataClass(
                    value=fields.get('DocumentNumber', {}).get('content'),
                    confidence=fields.get('DocumentNumber', {}).get('confidence'),
                ),
                last_name=ItemIdentityParserDataClass(
                    value=fields.get('LastName', {}).get('content'),
                    confidence=fields.get('LastName', {}).get('confidence'),
                ),
                given_names=final_given_names,
                mrz=ItemIdentityParserDataClass(
                    value=fields.get('MachineReadableZone', {}).get('content'),
                    confidence=fields.get('MachineReadableZone', {}).get('confidence'),
                ),
                nationality=ItemIdentityParserDataClass(
                    value=fields.get('Nationality', {}).get('content'),
                    confidence=fields.get('Nationality', {}).get('confidence'),
                ),
                birth_place=ItemIdentityParserDataClass(
                    value=fields.get('PlaceOfBirth', {}).get('content'),
                    confidence=fields.get('PlaceOfBirth', {}).get('confidence'),
                ),
                gender=ItemIdentityParserDataClass(
                    value=fields.get('Sex', {}).get('content'),
                    confidence=fields.get('Sex', {}).get('confidence'),
                )
            ))

        standarized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )

    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        :return:            VisionExplicitDetection Object that contains the
        the objects and their location
        """

        # Getting response of API
        response = requests.post(
            f"{self.url['vision']}/analyze?visualFeatures=Adult",
            headers=self.headers["vision"],
            data=file,
        )
        data = response.json()

        # error handling
        if response.status_code != 200:
            if response.status_code == 415:
                # 415 response doesn't have 'error' key
                raise ProviderException(data["message"])
            else:
                raise ProviderException(data["error"]["message"])

        # key is adult but contains all categories (gore, racy, adult)
        moderation_content = data["adult"]

        # Getting the explicit label and its score of image
        items = []
        for explicit_type in ["gore", "adult", "racy"]:
            if moderation_content.get(f"{explicit_type}Score"):
                items.append(
                    ExplicitItem(
                        label=explicit_type.capitalize(),
                        likelihood=content_processing(
                            moderation_content[f"{explicit_type}Score"]
                        ),
                    )
                )
        nsfw = ExplicitContentDataClass.calculate_nsfw_likelihood(items)

        res =  ResponseType[ExplicitContentDataClass](
            original_response=data,
            standarized_response=ExplicitContentDataClass(
                items=items, nsfw_likelihood=nsfw
            ),
        )
        print(res.dict())
        return res



    def image__object_detection(self,
        file: BufferedReader
    ) -> ResponseType[ObjectDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        :return:            VisionObjectDetection Object that contains the
        the objects and their location
        """
        # Call api
        response = requests.post(
            f"{self.url['vision']}/detect",
            headers=self.headers["vision"],
            data=file,
        ).json()

        items = []

        width, height = response["metadata"]["width"], response["metadata"]["height"]

        for obj in response["objects"]:
            items.append(
                ObjectItem(
                    label=obj["object"],
                    confidence=obj["confidence"],
                    x_min=obj["rectangle"]["x"] / width,
                    x_max=(obj["rectangle"]["x"] + obj["rectangle"]["w"]) / width,
                    y_min=1 - ((height - obj["rectangle"]["y"]) / height),
                    y_max=1
                    - (
                        (height - obj["rectangle"]["y"] - obj["rectangle"]["h"])
                        / height
                    ),
                )
            )

        return ResponseType[ObjectDetectionDataClass](
            original_response = response,
            standarized_response= ObjectDetectionDataClass(items=items)
        )



    def image__face_detection(self,
        file: BufferedReader
    ) -> ResponseType[FaceDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        :return:            VisionFaceDetection Object that contains the
        the objects and their location
        """

        file_content = file.read()
        # Getting size of image
        img_size = Img.open(file).size

        # Create params for returning face attribute
        params = {
            "returnFaceId": "true",
            "returnFaceLandmarks": "true",
            "returnFaceAttributes": ("age,gender,headPose,smile,facialHair,glasses,emotion,"
                                    "hair,makeup,occlusion,accessories,blur,exposure,noise"),
        }
        # Getting response of API
        response = requests.post(
            f"{self.url['face']}/detect",
            params=params,
            headers=self.headers["face"],
            data=file_content,
        ).json()

        # handle error
        if not isinstance(response, list) and response.get("error") is not None:
            print(response)
            raise ProviderException(
                f'Error calling Microsoft Api: {response["error"].get("message", "error 500")}'
            )
        # Create response VisionFaceDetection object

        faces_list : List = miscrosoft_normalize_face_detection_response(response, img_size)

        return ResponseType[FaceDetectionDataClass](
            original_response= response,
            standarized_response= FaceDetectionDataClass(items=faces_list)
        )



    def image__logo_detection(self,
        file: BufferedReader
    ) -> ResponseType[LogoDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        """
        response = requests.post(
            f"{self.url['vision']}/analyze?visualFeatures=Brands",
            headers=self.headers["vision"],
            data=file,
        )
        data = response.json()

        if response.status_code != 200:
            # sometimes no "error" key in repsonse
            # ref: https://westcentralus.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-2/operations/56f91f2e778daf14a499f21b
            error_msg = data.get("message", data.get("error", "message"))
            raise ProviderException(error_msg)

        items: Sequence[LogoItem] = []
        for key in data.get("brands"):
            x_cordinate = float(key.get("rectangle").get("x"))
            y_cordinate = float(key.get("rectangle").get("y"))
            height = float(key.get("rectangle").get("h"))
            weidth = float(key.get("rectangle").get("w"))
            vertices = []
            vertices.append(LogoVertice(x=x_cordinate, y=y_cordinate))
            vertices.append(LogoVertice(x=x_cordinate + weidth, y=y_cordinate))
            vertices.append(LogoVertice(x=x_cordinate + weidth, y=y_cordinate + height))
            vertices.append(LogoVertice(x=x_cordinate, y=y_cordinate + height))

            items.append(
                LogoItem(
                    description=key.get("name"),
                    score=key.get("confidence"),
                    bounding_poly=LogoBoundingPoly(vertices=vertices),
                )
            )

        return ResponseType[LogoDetectionDataClass](
            original_response=data, standarized_response=LogoDetectionDataClass(items=items)
        )


    def image__landmark_detection(self,
        file: BufferedReader
    ) -> ResponseType[LandmarkDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        """

        file_content = file.read()

        # Getting response of API
        response = requests.post(
            f"{self.url['vision']}analyze?details=Landmarks",
            headers=self.headers["vision"],
            data=file_content,
        ).json()
        items: Sequence[LandmarkItem] = []
        for key in response.get("categories"):
            for landmark in key.get("detail").get("landmarks"):
                if landmark.get("name") not in [item.description for item in items]:
                    items.append(
                        LandmarkItem(
                            description=landmark.get("name"),
                            confidence=landmark.get("confidence"),
                        )
                    )

        return ResponseType[LandmarkDetectionDataClass](
            original_response=response,
            standarized_response= LandmarkDetectionDataClass(items=items)
        )



    def text__sentiment_analysis(self,
        language: str,
        text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        """
        :param language:    String that contains language code
        :param text:        String that contains the text to analyse
        :return:            TextSentimentAnalysis Object that contains sentiments and their rates
        """
        # Call api
        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "SentimentAnalysis",
                    "parameters": {
                        "modelVersion": "latest",
                    },
                    "analysisInput": {
                        "documents": [{"id": "1", "language": language, "text": text}]
                    },
                },
            )
        except Exception as exc:
            raise ProviderException(f"Unexpected error! {sys.exc_info()[0]}") from exc

        data = response.json()
        self._check_microsoft_error(data)

        items: Sequence[SegmentSentimentAnalysisDataClass] = []

        # Getting the explicit label and its score of image
        default_dict = defaultdict(lambda: None)
        sentences = data.get('results',default_dict).get('documents',[default_dict])[0].get('sentences')
        if sentences : 
            for sentence in sentences:
                best_sentiment = {
                    'sentiment': None,
                    'rate': 0,
                }
                for sentiment, value in sentence['confidenceScores'].items():
                    if best_sentiment['rate'] < value:
                        best_sentiment['sentiment'] = sentiment
                        best_sentiment['rate'] = value

                items.append(
                    SegmentSentimentAnalysisDataClass(
                        segment=sentence['text'],
                        sentiment=best_sentiment['sentiment'],
                        sentiment_rate=best_sentiment['rate'],
                    )
                )

        best_general_sentiment = {
            'sentiment': None,
            'rate': 0
        }
        for sentiment, value in data['results']['documents'][0]['confidenceScores'].items():
            if best_general_sentiment['rate'] < value:
                    best_general_sentiment['sentiment'] = sentiment
                    best_general_sentiment['rate'] = value

        standarize = SentimentAnalysisDataClass(
            general_sentiment=best_general_sentiment['sentiment'],
            general_sentiment_rate=best_general_sentiment['rate'],
            items=items
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response= data,
            standarized_response= standarize
        )



    def _check_microsoft_error(self, data: Dict):
        if data.get("error", {}).get("message"):
            raise Exception(data["error"]["message"])

    def text__keyword_extraction(self,
        language:str,
        text:str
    ) -> ResponseType[KeywordExtractionDataClass]:
        """
        :param language:    String that contains language code
        :param text:        String that contains the text to analyse
        :return:            TextKeywordExtraction Object that contains the Key phrases
        """

        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "KeyPhraseExtraction",
                    "parameters": {"modelVersion": "latest"},
                    "analysisInput": {
                        "documents": [{"id": "1", "language": language, "text": text}]
                    },
                },
            )
        except Exception as exc:
            raise ProviderException(f"Unexpected error! {sys.exc_info()[0]}") from exc
        data = response.json()
        self._check_microsoft_error(data)

        items: Sequence[InfosKeywordExtractionDataClass] = []
        for key_phrase in data["results"]["documents"][0]["keyPhrases"]:
            items.append(
                InfosKeywordExtractionDataClass(keyword=key_phrase)
            )

        standarized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=data,
            standarized_response=standarized_response
        )



    def translation__language_detection(self,
        text
    ) -> ResponseType[LanguageDetectionDataClass]:
        """
        :param text:        String that contains input text
        :return:            String that contains output result
        """

        # Call api
        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "LanguageDetection",
                    "parameters": {"modelVersion": "latest"},
                    "analysisInput": {"documents": [{"id": "1", "text": text}]},
                },
            )
        except SyntaxError as exc:
            raise ProviderException("Microsoft API raised an error") from exc

        data = response.json()
        self._check_microsoft_error(data)

        items: Sequence[InfosLanguageDetectionDataClass] = []
        # Analysing response
        result = data["results"]["documents"]
        if len(result) > 0:
            for lang in result:
                items.append(
                    InfosLanguageDetectionDataClass(
                        language=lang["detectedLanguage"]["iso6391Name"],
                        confidence=lang["detectedLanguage"]["confidenceScore"],
                    )
                )

        standarized_response = LanguageDetectionDataClass(items=items)

        return ResponseType[LanguageDetectionDataClass] (
            original_response= data,
            standarized_response= standarized_response
        )



    def translation__automatic_translation(
        self,
        source_language: str,
        target_language: str,
        text:str
    ) ->ResponseType[AutomaticTranslationDataClass] :
        """
        :param source_language:    String that contains language name of origin text
        :param target_language:    String that contains language name of origin text
        :param text:        String that contains input text to translate
        :return:            String that contains output result
        """

        # Create configuration dictionnary

        url = format_string_url_language(
            self.url["translator"], source_language, "from", self.provider_name
        )
        url = format_string_url_language(url, target_language, "to", self.provider_name)

        body = [
            {
                "text": text,
            }
        ]
        # Getting response of API
        response = requests.post(url, headers=self.headers["translator"], json=body)
        data = response.json()

        # Create output TextAutomaticTranslation object
        standarized_response = AutomaticTranslationDataClass(
            text=data[0]["translations"][0]["text"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response= data,
            standarized_response= standarized_response
        )



    def text__named_entity_recognition(self,
        language:str,
        text:str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        :return:                TextNamedEntityRecognition Object that contains
        the entities and their importances
        """

        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "EntityRecognition",
                    "parameters": {"modelVersion": "latest"},
                    "analysisInput": {
                        "documents": [{"id": "1", "language": language, "text": text}]
                    },
                },
            )
        except Exception as exc:
            raise ProviderException(f"Unexpected error! {sys.exc_info()[0]}") from exc

        data = response.json()
        self._check_microsoft_error(data)

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []
        for ent in data["results"]["documents"][0]["entities"]:
            entity = ent["text"]
            importance = ent["confidenceScore"]
            entity_type = ent["category"].upper()
            if entity_type == 'DATETIME':
                entity_type = 'DATE'
            
            items.append(
                InfosNamedEntityRecognitionDataClass(
                    entity=entity,
                    importance=importance,
                    category=entity_type,
                )
            )

        standarized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response= data,
            standarized_response= standarized_response
        )



    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: str = None
    ) -> ResponseType[SummarizeDataClass]:

        """
        :param text:        String that contains input text
        :return:            String that contains output result
        """

        response = requests.post(
            self.url["summarization"],
            headers=self.headers["text"],
            json={
                "analysisInput": {"documents": [{"id": "1", "text": text}]},
                "tasks": {
                    "extractiveSummarizationTasks": [
                        {
                            "parameters": {
                                "model-version": "latest",
                                "sentenceCount": output_sentences,
                                "sortBy": "Offset",
                            }
                        }
                    ]
                },
            },
        )
        get_url = response.headers["operation-location"]
        resp = requests.get(url=get_url, headers=self.headers["text"])
        data = resp.json()
        wait_time = 0
        while wait_time < 60:  # Wait for the answer from provider
            if data["status"] == "succeeded":
                sentences = data["tasks"]["extractiveSummarizationTasks"][0]["results"][
                    "documents"
                ][0]["sentences"]
                summary = " ".join([sentence["text"] for sentence in sentences])
                break
            time.sleep(6)
            wait_time += 6
            resp = requests.get(url=get_url, headers=self.headers["text"])
            data = resp.json()

        standarized_response = SummarizeDataClass(result=summary)

        return ResponseType[SummarizeDataClass](
            original_response= data,
            standarized_response= standarized_response
        )



    def ocr__ocr_tables_async__launch_job(self,
        file: BufferedReader,
        file_type: str,
        language: str
    ) -> AsyncLaunchJobResponseType:

        file_content = file.read()
        url = (f"{self.url['ocr_tables_async']}formrecognizer/documentModels/"
                f"prebuilt-layout:analyze?api-version=2022-01-30-preview")
        url = format_string_url_language(url, language, "locale", self.provider_name)

        response = requests.post(
            url, headers=self.headers["ocr_tables_async"], data=file_content
        )

        return AsyncLaunchJobResponseType(provider_job_id=response.headers.get("apim-request-id"))

    def ocr__ocr_tables_async__get_job_result(self,
        job_id: str
    ) -> AsyncBaseResponseType[OcrTablesAsyncDataClass]:

        headers = self.headers["ocr_tables_async"]
        url = (
            self.url["ocr_tables_async"]
            + f"formrecognizer/documentModels/prebuilt-layout/"
            f"analyzeResults/{job_id}?api-version=2022-01-30-preview"
        )
        response = requests.get(url, headers=headers)
        data = response.json()
        if data.get("error"):
            raise ProviderException(data.get("error"))
        if data["status"] == "succeeded":
            original_result = data["analyzeResult"]

            def microsoft_async(original_result):
                page_num = 1
                pages: Sequence[Page] = []
                num_pages = len(original_result["pages"])
                tables: Sequence[Table] = []
                for table in original_result["tables"]:
                    rows: Sequence[Row] = []
                    num_rows = table["rowCount"]
                    num_cols = table["columnCount"]
                    row_index = 0
                    cells: Sequence[Cell] = []
                    all_header = True  # all cells in a row are "header" kind
                    is_header = False
                    for cell in table["cells"]:
                        bounding_box = cell["boundingRegions"][0]["boundingBox"]
                        width = original_result["pages"][page_num - 1]["width"]
                        height = original_result["pages"][page_num - 1]["height"]

                        ocr_cell = Cell(
                            text=cell["content"],
                            row_span=cell["rowSpan"],
                            col_span=cell["columnSpan"],
                            bounding_box=BoundixBoxOCRTable(
                                height=(bounding_box[7] - bounding_box[3]) / height,
                                width=(bounding_box[2] - bounding_box[0]) / width,
                                left=bounding_box[1] / width,
                                top=bounding_box[0] / height,
                            ),
                        )
                        if (
                            "kind" not in cell.keys()
                            or "kind" in cell.keys()
                            and "Header" not in cell["kind"]
                        ):
                            all_header = False
                        current_row_index = cell["rowIndex"]
                        if current_row_index > row_index:
                            row_index = current_row_index
                            if all_header:
                                is_header = True
                            all_header = True
                            rows.append(Row(is_header=is_header, cells=cells))
                            cells = []
                        cells.append(ocr_cell)
                        current_page_num = cell["boundingRegions"][0]["pageNumber"]
                        if current_page_num > page_num:
                            page_num = current_page_num
                            pages.append(Page())
                            tables = []
                    ocr_table = Table(rows=rows, num_cols=num_cols, num_rows=num_rows)
                    tables.append(ocr_table)
                    ocr_page = Page(tables=tables)
                pages.append(ocr_page)
                standarized_response = OcrTablesAsyncDataClass(
                    pages=pages, num_pages=num_pages
                )
                return standarized_response.dict()

            standarized_response = microsoft_async(original_result)
            return AsyncResponseType[OcrTablesAsyncDataClass](
                original_response= data,
                standarized_response= standarized_response,
                provider_job_id= job_id
            )

        return AsyncPendingResponseType[OcrTablesAsyncDataClass](
            provider_job_id= job_id
        )


    def audio__speech_to_text_async__launch_job(
        self,
        file: BufferedReader,
        language: str, speakers : int, profanity_filter: bool,
        vocabulary: list
    ) -> AsyncLaunchJobResponseType:

        #check language
        if not language:
            raise LanguageException("Language not provided")
        wav_file, *_options = wav_converter(file, channels=1)
        content_url = upload_file_to_s3(wav_file, Path(file.name).stem + ".wav")

        headers = self.headers["speech"]
        headers["Content-Type"] = "application/json"

        config = {
            "contentUrls": [content_url],
            "properties": {
                "wordLevelTimestampsEnabled": True,
                "diarizationEnabled": True,
                "profanityFilterMode" : "None"
            },
            "locale": language,
            "displayName": "test batch transcription",
        }
        if profanity_filter:
            config["properties"].update({
                "profanityFilterMode" : "Masked"
            })
        # if not profanity_filter:
        #     config["properties"]["profanityFilterMode"] = "Removed"

        response = requests.post(
            url=self.url["speech"], headers=headers, data=json.dumps(config)
        )
        print(response.json())
        if response.status_code == 201:
            result_location = response.headers["Location"]
            provider_id = result_location.split("/")[-1]
            return AsyncLaunchJobResponseType(provider_job_id=provider_id)
        else:
            raise Exception(response.json().get("message"))

    def audio__speech_to_text_async__get_job_result(self,
        provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:

        headers = self.headers["speech"]
        response = requests.get(
            url=f'{self.url["speech"]}/{provider_job_id}/files', headers=headers
        )
        print(response.json())
        original_response=None
        if response.status_code == 200:
            data = response.json()["values"]
            if data:
                files_urls = [
                    entry["links"]["contentUrl"]
                    for entry in data
                    if entry["kind"] == "Transcription"
                ]
                text = ""
                diarization_entries = []
                speakers = set()
                for file_url in files_urls:
                    response = requests.get(file_url, headers=headers)
                    original_response = response.json()
                    if response.status_code != 200:
                        error = original_response.get("message")
                        raise ProviderException(error)

                    data = original_response["combinedRecognizedPhrases"][0]
                    text += data["display"]
                    for recognized_status in original_response["recognizedPhrases"]:
                        if recognized_status["recognitionStatus"] == "Success":
                            speaker = recognized_status["speaker"]
                            for word_info in recognized_status["nBest"][0]["words"]:
                                speakers.add(speaker)
                                diarization_entries.append(
                                    SpeechDiarizationEntry(
                                        segment= word_info["word"],
                                        speaker=speaker,
                                        start_time= word_info["offset"].split('PT')[1][:-1],
                                        end_time= str(float(word_info["offset"].split('PT')[1][:-1])+ float(word_info["duration"].split('PT')[1][:-1])),
                                        confidence= float(word_info["confidence"])
                                    )
                                )
                diarization = SpeechDiarization(total_speakers=len(speakers), entries= diarization_entries)

                standarized_response = SpeechToTextAsyncDataClass(text=text, diarization=diarization)
                return AsyncResponseType[SpeechToTextAsyncDataClass](
                    original_response= original_response,
                    standarized_response= standarized_response,
                    provider_job_id= provider_job_id
                )
            else:
                return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                    provider_job_id=provider_job_id
                )
        else:
            error = response.json().get("message")
            raise ProviderException(error)
