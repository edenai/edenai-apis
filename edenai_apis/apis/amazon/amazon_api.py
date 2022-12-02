import datetime
import json
from io import BufferedReader, BytesIO
from pprint import pprint
from time import time
from typing import Sequence
import base64
import uuid

import urllib
from pathlib import Path
from pdf2image.pdf2image import convert_from_bytes
from PIL import Image as Img

from edenai_apis.features.base_provider.provider_api import ProviderApi
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.apis.amazon.helpers import content_processing
from edenai_apis.features import Audio, Video, Text, Image, Ocr, Translation
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    TextToSpeechDataClass,
    SpeechDiarization,
    SpeechDiarizationEntry
)
from edenai_apis.features.ocr import (
    OcrTablesAsyncDataClass,
    Bounding_box,
    OcrDataClass,
    InfosIdentityParserDataClass,
    InfoCountry,
    get_info_country,
)
from edenai_apis.features.image import (
    ObjectItem,
    ObjectDetectionDataClass,
    FaceFeatures,
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceFacialHair,
    FaceItem,
    FaceLandmarks,
    FaceQuality,
    FacePoses,
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import IdentityParserDataClass, ItemIdentityParserDataClass, format_date
from edenai_apis.features.text import (
    InfosKeywordExtractionDataClass,
    KeywordExtractionDataClass,
    SentimentAnalysisDataClass,
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
    InfosSyntaxAnalysisDataClass,
    SyntaxAnalysisDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import SegmentSentimentAnalysisDataClass
from edenai_apis.features.translation import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
    AutomaticTranslationDataClass,
)
from edenai_apis.features.video import (
    FaceAttributes,
    FaceDetectionAsyncDataClass,
    LandmarksVideo,
    VideoBoundingBox,
    VideoFace,
    VideoFacePoses,
    LabelDetectionAsyncDataClass,
    VideoLabel,
    VideoLabelBoundingBox,
    VideoLabelTimeStamp,
    TextDetectionAsyncDataClass,
    VideoText,
    VideoTextBoundingBox,
    VideoTextFrames,
    ContentNSFW,
    ExplicitContentDetectionAsyncDataClass,
    PersonTracking,
    PersonTrackingAsyncDataClass,
    VideoTrackingBoundingBox,
    VideoTrackingPerson,
)
from edenai_apis.utils.audio import wav_converter
from edenai_apis.utils.exception import (
    ProviderException,
    LanguageException
)
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncBaseResponseType,
    ResponseType,
    AsyncResponseType,
)

from .config import clients, audio_voices_ids, tags, storage_clients

from .helpers import (
    check_webhook_result,
    amazon_launch_video_job,
    amazon_video_response_formatter,
    amazon_ocr_tables_parser
)

from botocore.exceptions import ClientError, ParamValidationError

class AmazonApi(
    ProviderApi,
    Image,
    Ocr,
    Text,
    Translation,
    Video,
    Audio,
):
    provider_name = "amazon"
    
    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
        self.clients = clients(self.api_settings)
        self.storage_clients = storage_clients(self.api_settings)
        
    def image__object_detection(
        self, file: BufferedReader
    ) -> ResponseType[ObjectDetectionDataClass]:

        file_content = file.read()
        # Getting API response
        original_response = self.clients["image"].detect_labels(
            Image={"Bytes": file_content}, MinConfidence=70
        )
        # Standarization
        items = []
        for object_label in original_response.get("Labels"):

            if object_label.get("Instances"):
                bounding_box = object_label.get("Instances")[0].get("BoundingBox")
                x_min, x_max = (
                    bounding_box.get("Left"),
                    bounding_box.get("Left") + bounding_box.get("Width"),
                )
                y_min, y_max = (
                    bounding_box.get("Top"),
                    bounding_box.get("Top") + bounding_box.get("Height"),
                )
            else:
                x_min, x_max, y_min, y_max = None, None, None, None

            items.append(
                ObjectItem(
                    label=object_label.get("Name"),
                    confidence=object_label.get("Confidence") / 100,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

        return ResponseType[ObjectDetectionDataClass](
            original_response=original_response,
            standarized_response=ObjectDetectionDataClass(items=items),
        )

    def image__face_detection(
        self, file: BufferedReader
    ) -> ResponseType[FaceDetectionDataClass]:
        file_content = file.read()

        # Getting Response
        original_response = self.clients["image"].detect_faces(
            Image={"Bytes": file_content}, Attributes=["ALL"]
        )

        # Standarize Response
        faces_list = []
        for face in original_response.get("FaceDetails", []):

            # Age
            age_output = None
            age_range = face.get("AgeRange")
            if age_range:
                age_output = (
                    age_range.get("Low", 0.0) + age_range.get("High", 100)
                ) / 2

            # features
            features = FaceFeatures(
                eyes_open=face.get("eyes_open", {}).get("Confidence", 0.0) / 100,
                smile=face.get("smile", {}).get("Confidence", 0.0) / 100,
                mouth_open=face.get("mouth_open", {}).get("Confidence", 0.0) / 100,
            )

            # accessories
            accessories = FaceAccessories(
                sunglasses=face.get("Sunglasses", {}).get("Confidence", 0.0) / 100,
                eyeglasses=face.get("Eyeglasses", {}).get("Confidence", 0.0) / 100,
            )

            # facial hair
            facial_hair = FaceFacialHair(
                moustache=face.get("Mustache", {}).get("Confidence", 0.0) / 100,
                beard=face.get("Beard", {}).get("Confidence", 0.0) / 100,
            )

            # quality
            quality = FaceQuality(
                brightness=face.get("Quality").get("Brightness", 0.0) / 100,
                sharpness=face.get("Quality").get("Sharpness", 0.0) / 100,
            )

            # emotions
            emotion_output = {}
            for emo in face.get("Emotions", []):
                normalized_emo = emo.get("Confidence", 0.0) * 100
                if emo.get("Type"):
                    if emo.get("Type").lower() == "happy":  # normalise keywords
                        emo["Type"] = "happiness"
                    emotion_output[emo.get("Type").lower()] = content_processing(
                        normalized_emo
                    )
            emotions = FaceEmotions(
                anger=emotion_output.get("angry"),
                surprise=emotion_output.get("surprise"),
                fear=emotion_output.get("fear"),
                sorrow=emotion_output.get("sadness"),
                confusion=emotion_output.get("confused"),
                calm=emotion_output.get("calm"),
                disgust=emotion_output.get("disgsusted"),
                joy=emotion_output.get("happiness"),
            )

            # landmarks
            landmarks_output = {}
            for land in face.get("Landmarks"):
                if land.get("Type") and land.get("X") and land.get("Y"):
                    landmarks_output[land.get("Type")] = [land.get("X"), land.get("Y")]

            landmarks = FaceLandmarks(
                left_eye=landmarks_output.get("eye_left", []),
                left_eye_top=landmarks_output.get("eye_leftUp", []),
                left_eye_right=landmarks_output.get("lefteye_right", []),
                left_eye_bottom=landmarks_output.get("leftEyeDown", []),
                left_eye_left=landmarks_output.get("leftEyeLeft", []),
                right_eye=landmarks_output.get("eye_right", []),
                right_eye_top=landmarks_output.get("eye_rightUp", []),
                right_eye_right=landmarks_output.get("eye_rightRight", []),
                right_eye_bottom=landmarks_output.get("rightEyeDown", []),
                right_eye_left=landmarks_output.get("rightEyeLeft", []),
                left_eyebrow_left=landmarks_output.get("leftEyeBrowLeft", []),
                left_eyebrow_right=landmarks_output.get("leftEyeBrowRight", []),
                left_eyebrow_top=landmarks_output.get("leftEyeBrowUp", []),
                right_eyebrow_left=landmarks_output.get("rightEyeBrowLeft", []),
                right_eyebrow_right=landmarks_output.get("rightEyeBrowRight", []),
                right_eyebrow_top=landmarks_output.get("rightEyeBrowUp", []),
                left_pupil=landmarks_output.get("leftPupil", []),
                right_pupil=landmarks_output.get("rightPupil", []),
                nose_tip=landmarks_output.get("nose", []),
                nose_bottom_right=landmarks_output.get("noseRight", []),
                nose_bottom_left=landmarks_output.get("noseLeft", []),
                mouth_left=landmarks_output.get("mouth_left", []),
                mouth_right=landmarks_output.get("mouth_right", []),
                mouth_top=landmarks_output.get("mouthUp", []),
                mouth_bottom=landmarks_output.get("mouthDown", []),
                chin_gnathion=landmarks_output.get("chinBottom", []),
                upper_jawline_left=landmarks_output.get("upperJawlineLeft", []),
                mid_jawline_left=landmarks_output.get("midJawlineLeft", []),
                mid_jawline_right=landmarks_output.get("midJawlineRight", []),
                upper_jawline_right=landmarks_output.get("upperJawlineRight", []),
            )
            poses = FacePoses(
                roll=face.get("Pose", {}).get("Roll"),
                yaw=face.get("Pose", {}).get("Yaw"),
                pitch=face.get("Pose", {}).get("Pitch"),
            )

            faces_list.append(
                FaceItem(
                    age=age_output,
                    gender=face.get("Gender", {}).get("Value"),
                    facial_hair=facial_hair,
                    features=features,
                    accessories=accessories,
                    quality=quality,
                    emotions=emotions,
                    landmarks=landmarks,
                    poses=poses,
                    confidence=face.get("Confidence", 0.0) / 100,
                    bounding_box=FaceBoundingBox(
                        x_min=face.get("BoundingBox", {}).get("Left", 0.0),
                        x_max=face.get("BoundingBox", {}).get("Left", 0.0)
                        + face.get("BoundingBox", {}).get("Width", 0.0),
                        y_min=face.get("BoundingBox", {}).get("Top", 0.0),
                        y_max=face.get("BoundingBox", {}).get("Top", 0.0)
                        + face.get("BoundingBox", {}).get("Height", 0.0),
                    ),
                )
            )

        standarized_response = FaceDetectionDataClass(items=faces_list)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )

    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        file_content = file.read()
        response = self.clients["image"].detect_moderation_labels(
            Image={"Bytes": file_content}, MinConfidence=20
        )

        items = []
        for label in response.get("ModerationLabels"):
            items.append(
                ExplicitItem(
                    label=label.get("Name"),
                    likelihood=content_processing(label.get("Confidence")),
                )
            )

        standarized_response = ExplicitContentDataClass(items=items)

        return ResponseType[ExplicitContentDataClass](
            original_response=response, standarized_response=standarized_response
        )

    def ocr__ocr(
        self, file: BufferedReader, language: str
    ) -> ResponseType[OcrDataClass]:
        file_content = file.read()

        try:
            response = self.clients["textract"].detect_document_text(
                Document={
                    "Bytes": file_content,
                    "S3Object": {"Bucket": api_settings["bucket"], "Name": file.name},
                }
            )
        except Exception as amazon_call_exception:
            raise ProviderException(str(amazon_call_exception))

        final_text = ""
        output_value = json.dumps(response, ensure_ascii=False)
        original_response = json.loads(output_value)
        boxes: Sequence[Bounding_box] = []

        # Get region of text
        for region in original_response.get("Blocks"):
            if region.get("BlockType") == "LINE":
                # Read line by region
                final_text += " " + region.get("Text")

            if region.get("BlockType") == "WORD":
                boxes.append(
                    Bounding_box(
                        text=region.get("Text"),
                        left=region["Geometry"]["BoundingBox"]["Left"],
                        top=region["Geometry"]["BoundingBox"]["Top"],
                        width=region["Geometry"]["BoundingBox"]["Width"],
                        height=region["Geometry"]["BoundingBox"]["Height"],
                    )
                )

        standardized = OcrDataClass(
            text=final_text.replace("\n", " ").strip(), bounding_boxes=boxes
        )

        return ResponseType[OcrDataClass](
            original_response=original_response, standarized_response=standardized
        )

    def ocr__identity_parser(self, file: BufferedReader, filename: str) -> ResponseType[IdentityParserDataClass]:
        original_response = self.clients.get('textract').analyze_id(DocumentPages=[{
            "Bytes": file.read(),
            "S3Object": { 'Bucket': api_settings['bucket'], 'Name': filename}
        }])

        items = []
        for document in original_response['IdentityDocuments']:
            infos = {}
            infos['given_names'] = []
            for field in document['IdentityDocumentFields']:
                field_type = field['Type']['Text']
                confidence = round(field['ValueDetection']['Confidence'] / 100, 2)
                value = field['ValueDetection']['Text'] if field['ValueDetection']['Text'] != "" else None
                if field_type == 'LAST_NAME':
                    infos['last_name'] = ItemIdentityParserDataClass(
                        value=value,
                        confidence=confidence
                        )
                elif field_type in ('FIRST_NAME', 'MIDDLE_NAME') and value:
                    infos['given_names'].append(ItemIdentityParserDataClass(
                        value=value,
                        confidence=confidence
                        ))
                elif field_type == 'DOCUMENT_NUMBER':
                    infos['document_id'] = ItemIdentityParserDataClass(
                        value=value,
                        confidence=confidence
                        )
                elif field_type == 'EXPIRATION_DATE':
                    value = field['ValueDetection'].get('NormalizedValue', {}).get('Value')
                    infos['expire_date'] = ItemIdentityParserDataClass(
                        value=format_date(value, '%Y-%m-%dT%H:%M:%S'),
                        confidence=confidence
                        )
                elif field_type == 'DATE_OF_BIRTH':
                    value = field['ValueDetection'].get('NormalizedValue', {}).get('Value')
                    infos['birth_date'] = ItemIdentityParserDataClass(
                        value=format_date(value, '%Y-%m-%dT%H:%M:%S'),
                        confidence=confidence
                        )
                elif field_type == 'DATE_OF_ISSUE':
                    value = field['ValueDetection'].get('NormalizedValue', {}).get('Value')
                    infos['issuance_date'] = ItemIdentityParserDataClass(
                        value=format_date(value, '%Y-%m-%dT%H:%M:%S'),
                        confidence=confidence
                        )
                elif field_type == 'ID_TYPE':
                    infos['document_type'] = ItemIdentityParserDataClass(
                        value=value,
                        confidence=confidence
                        )
                elif field_type == 'ADDRESS':
                    infos['address'] = ItemIdentityParserDataClass(
                        value=value,
                        confidence=confidence
                        )
                elif field_type == 'COUNTY' and value:
                    infos['country'] = get_info_country(InfoCountry.NAME, value)
                    infos['country']['confidence'] = confidence
                elif field_type == 'MRZ_CODE':
                    infos['mrz'] = ItemIdentityParserDataClass(value=value, confidence=confidence)

            items.append(infos)

            pprint(items)

        standarized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )


    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        # Getting response
        try:
            response = self.clients["text"].detect_sentiment(
                Text=text, LanguageCode=language
            )
        except ClientError as exc:
            if "languageCode" in str(exc):
                raise LanguageException(str(exc))

        # Analysing response

        best_sentiment = {
            "general_sentiment": None,
            "general_sentiment_rate": 0,
            "items": []
        }
        
        for key in response["SentimentScore"]:
            if key == 'Mixed':
                continue

            if best_sentiment['general_sentiment_rate'] <= response["SentimentScore"][key]:
                best_sentiment["general_sentiment"] = key
                best_sentiment['general_sentiment_rate'] = response["SentimentScore"][key]

        standarize = SentimentAnalysisDataClass(
            general_sentiment=best_sentiment['general_sentiment'],
            general_sentiment_rate=best_sentiment['general_sentiment_rate'],
            items=[]
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=response, standarized_response=standarize
        )

    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        # Getting response
        try:
            response = self.clients["text"].detect_key_phrases(
                Text=text, LanguageCode=language
            )
        except ClientError as exc:
            if "languageCode" in str(exc):
                raise LanguageException(str(exc))

        # Analysing response
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for key_phrase in response["KeyPhrases"]:
            items.append(
                InfosKeywordExtractionDataClass(
                    keyword=key_phrase["Text"], importance=key_phrase["Score"]
                )
            )

        standarized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=response, standarized_response=standarized_response
        )

    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        # Getting response
        try:
            response = self.clients["text"].detect_entities(Text=text, LanguageCode=language)
        except ClientError as exc:
            if "languageCode" in str(exc):
                raise LanguageException(str(exc))

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []
        for ent in response["Entities"]:

            items.append(
                InfosNamedEntityRecognitionDataClass(
                    entity=ent["Text"],
                    importance=ent["Score"],
                    category=ent["Type"],
                )
            )

        standardized = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=response, standarized_response=standardized
        )

    def text__syntax_analysis(
        self, language: str, text: str
    ) -> ResponseType[SyntaxAnalysisDataClass]:

        # Getting response
        try:
            response = self.clients["text"].detect_syntax(Text=text, LanguageCode=language)
        except ClientError as exc:
            if "languageCode" in str(exc):
                raise LanguageException(str(exc))

        # Create output TextSyntaxAnalysis object

        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        # Analysing response
        #
        # Getting syntax detected of word and its score of confidence
        for ent in response["SyntaxTokens"]:
            tag = tags[ent["PartOfSpeech"]["Tag"]]
            items.append(
                InfosSyntaxAnalysisDataClass(
                    word=ent["Text"],
                    importance=ent["PartOfSpeech"]["Score"],
                    tag=tag,
                )
            )

        standarized_response = SyntaxAnalysisDataClass(items=items)

        return ResponseType[SyntaxAnalysisDataClass](
            original_response=response, standarized_response=standarized_response
        )

    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        response = self.clients["text"].detect_dominant_language(Text=text)

        # Create output TextDetectLanguage object
        # Analyze response
        # Getting the language's code detected and its score of confidence
        items: Sequence[InfosLanguageDetectionDataClass] = []
        if len(response["Languages"]) > 0:
            for lang in response["Languages"]:
                items.append(
                    InfosLanguageDetectionDataClass(
                        language=lang["LanguageCode"], confidence=lang["Score"]
                    )
                )

        standarized_response = LanguageDetectionDataClass(items=items)

        return ResponseType[LanguageDetectionDataClass](
            original_response=response, standarized_response=standarized_response
        )

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        try:
            response = self.clients["translate"].translate_text(
                Text=text,
                SourceLanguageCode=source_language,
                TargetLanguageCode=target_language,
            )
        except ParamValidationError as exc:
            if "SourceLanguageCode" in str(exc):
                raise LanguageException(str(exc))

        standardized: AutomaticTranslationDataClass
        if response["TranslatedText"] != "":
            standardized = AutomaticTranslationDataClass(text=response["TranslatedText"])

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response, standarized_response=standardized.dict()
        )

    def audio__text_to_speech(
        self, language: str, text: str, option: str
    ) -> ResponseType[TextToSpeechDataClass]:

        formated_language = language
        voiceid = audio_voices_ids[formated_language][option]

        response = self.clients["texttospeech"].synthesize_speech(
            VoiceId=voiceid, OutputFormat="mp3", Text=text
        )

        # convert 'StreamBody' to b64
        audio_file = base64.b64encode(response["AudioStream"].read()).decode("utf-8")
        voice_type = 1

        standarized_response = TextToSpeechDataClass(
            audio=audio_file, voice_type=voice_type
        )
        return ResponseType[TextToSpeechDataClass](
            original_response={}, standarized_response=standarized_response
        )

    def ocr__ocr_tables_async__launch_job(
        self, file: BufferedReader, file_type: str, language: str
    ) -> AsyncLaunchJobResponseType:
        file_content = file.read()

        # upload file first
        self.storage_clients["textract"].Bucket(api_settings['bucket']).put_object(
            Key=file.name, Body=file_content
        )

        response = self.clients["textract"].start_document_analysis(
            DocumentLocation={
                "S3Object": {"Bucket": api_settings['bucket'], "Name": file.name},
            },
            FeatureTypes=[
                "TABLES",
            ],
            NotificationChannel={
                "SNSTopicArn": api_settings['topic'],
                "RoleArn": api_settings['role'],
            },
        )

        return AsyncLaunchJobResponseType(
            provider_job_id=response["JobId"]
        )

    def ocr__ocr_tables_async__get_job_result(
        self, job_id: str
    ) -> AsyncBaseResponseType[OcrTablesAsyncDataClass]:
        # Getting results from webhook.site
        data = check_webhook_result(job_id, self.api_settings)
        if data is None :
            return AsyncPendingResponseType[OcrTablesAsyncDataClass](
                provider_job_id=job_id
            )

        msg = json.loads(data.get("Message"))
        # ref: https://docs.aws.amazon.com/textract/latest/dg/async-notification-payload.html
        job_id = msg["JobId"]

        if msg["Status"] == "SUCCEEDED":
            original_result = self.clients["textract"].get_document_analysis(JobId=job_id)

            standarized_response = amazon_ocr_tables_parser(original_result)
            return AsyncResponseType[OcrTablesAsyncDataClass](
                original_response=original_result,
                standarized_response=standarized_response,
                provider_job_id=job_id,
            )
        elif msg["Status"] == "PROCESSING":
            return AsyncPendingResponseType[OcrTablesAsyncDataClass](provider_job_id=job_id)

        else:
            original_result = self.clients["textract"].get_document_analysis(JobId=job_id)
            if original_result.get("JobStatus") == "FAILED":
                error = original_result.get("StatusMessage")
                raise ProviderException(error)


    # Speech to text async
    def _upload_audio_file_to_amazon_server(
        self, file: BufferedReader, file_name: str
    ) -> str:
        """
        :param audio_path:  String that contains the audio file path
        :return:            String that contains the filename on the server
        """
        # Store file in an Amazon server
        filename = str(int(time())) + "_" + str(file_name)
        self.storage_clients["speech"].meta.client.upload_fileobj(file, self.api_settings['bucket'], filename)

        return filename

    def _create_vocabulary(self, language:str, list_vocabs: list):
        list_vocabs = ["-".join(vocab.strip().split()) for vocab in list_vocabs]
        vocab_name = str(uuid.uuid4())
        try:
            clients["speech"].create_vocabulary(
            LanguageCode = language,
            VocabularyName = vocab_name,
            Phrases = list_vocabs
        )
        except Exception as exc:
            raise ProviderException(str(exc)) from exc

        return vocab_name

    def _launch_transcribe(
        self, filename:str, frame_rate, 
        language:str, speakers: int, vocab_name:str=None,
        initiate_vocab:bool= False):
        params = {
            "TranscriptionJobName" : filename,
            "Media" : {"MediaFileUri": self.api_settings["storage_url"] + filename},
            "MediaFormat" : "wav",
            "LanguageCode" : language,
            "MediaSampleRateHertz" : frame_rate,
            "Settings" : {
                "ShowSpeakerLabels": True,
                "ChannelIdentification": False,
                "MaxSpeakerLabels" : speakers
            }
        }
        if not language:
            del params["LanguageCode"]
            params.update({
                "IdentifyLanguage" : True
            })
        if vocab_name:
            params["Settings"].update({
                "VocabularyName": vocab_name
            })
            if initiate_vocab:
                params["checked"]= False
                extention_index = filename.rfind(".")
                filename = f"{filename[:-(len(filename) - extention_index)]}_settings.txt"
                storage_clients["speech"].meta.client.put_object(
                    Bucket=api_settings['bucket'], 
                    Body=json.dumps(params).encode(), 
                    Key=filename
                )
                return 
        try:
            clients["speech"].start_transcription_job(**params)
        except KeyError as exc:
            raise ProviderException(str(exc)) from exc


    def audio__speech_to_text_async__launch_job(
        self, file: BufferedReader, language: str, speakers : int,
        profanity_filter: bool, vocabulary: list
    ) -> AsyncLaunchJobResponseType:
        # Convert audio file in wav
        wav_file, frame_rate = wav_converter(file)[0:2]
        filename = self._upload_audio_file_to_amazon_server(
            wav_file, Path(file.name).stem + ".wav"
        )
        if vocabulary:
            vocab_name = self._create_vocabulary(language, vocabulary)
            self._launch_transcribe(filename, frame_rate, language, speakers, vocab_name, True)
            return AsyncLaunchJobResponseType(
                provider_job_id=f"{filename}EdenAI{vocab_name}"
            )

        self._launch_transcribe(filename, frame_rate, language, speakers)
        return AsyncLaunchJobResponseType(
            provider_job_id=filename
        )

        

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:

        # check custom vocabilory job state
        job_id, *vocab = provider_job_id.split("EdenAI")
        if vocab: # if vocabilory is used and
            setting_content = storage_clients["speech"].meta.client.get_object(Bucket= api_settings['bucket'], Key= f"{job_id[:-4]}_settings.txt")
            settings = json.loads(setting_content['Body'].read().decode('utf-8'))
            if not settings["checked"]: # check if the vocabulary has been created or not
                vocab_name = vocab[0]
                job_vocab_details = clients["speech"].get_vocabulary(VocabularyName = vocab_name)
                if job_vocab_details['VocabularyState'] == "FAILED":
                    error = job_vocab_details.get("FailureReason")
                    raise ProviderException(error)
                if job_vocab_details['VocabularyState'] != "READY":
                    return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                        provider_job_id=provider_job_id
                    )
                self._launch_transcribe(
                            settings['TranscriptionJobName'],
                            settings['MediaSampleRateHertz'], 
                            settings['LanguageCode'], 
                            settings['Settings']['MaxSpeakerLabels'], 
                            settings['Settings']['VocabularyName']
                )
                settings["checked"] = True # conform vocabulary creation
                extention_index = job_id.rfind(".")
                index_last = len(job_id) - extention_index
                storage_clients["speech"].meta.client.put_object(
                    Bucket=self.api_settings['bucket'], 
                    Body=json.dumps(settings).encode(),
                    Key = f"{job_id[:-index_last]}_settings.txt"
                )
                return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                        provider_job_id=provider_job_id
                    )

        #check transcribe status
        job_details = clients["speech"].get_transcription_job(
            TranscriptionJobName=job_id
        )
        job_status = job_details["TranscriptionJob"]["TranscriptionJobStatus"]
        if job_status == "COMPLETED":
            #delete vocabulary
            try:
                clients["speech"].delete_vocabulary(
                    VocabularyName = vocab[0]
                )
            except IndexError as ir: # if not vocabulary was created
                pass
            except Exception as exc:
                raise ProviderException(str(exc)) from exc
            json_res = job_details["TranscriptionJob"]["Transcript"][
                "TranscriptFileUri"
            ]
            with urllib.request.urlopen(json_res) as url:
                original_response = json.loads(url.read().decode("utf-8"))
                #diarization
                diarization_entries = []
                words_info = original_response["results"]["items"]
                speakers = original_response["results"]["speaker_labels"]["speakers"]

                for word_info in words_info:
                    if word_info.get('speaker_label'):
                        if word_info["type"] == "pronunciation":
                            diarization_entries.append(
                                SpeechDiarizationEntry(
                                    segment= word_info["alternatives"][0]["content"],
                                    speaker= int(word_info['speaker_label'].split("spk_")[1])+1,
                                    start_time= word_info['start_time'],
                                    end_time= word_info['end_time'],
                                    confidence= word_info["alternatives"][0]["confidence"]
                                )
                            )
                        else:
                            diarization_entries[len(diarization_entries)-1].segment = (
                                f"{diarization_entries[len(diarization_entries)-1].segment}"
                                f"{word_info['alternatives'][0]['content']}"
                            )

                standarized_response = SpeechToTextAsyncDataClass(
                    text=original_response["results"]["transcripts"][0]["transcript"],
                    diarization= SpeechDiarization(total_speakers=speakers, entries=diarization_entries)
                )
                return AsyncResponseType[SpeechToTextAsyncDataClass](
                    original_response=original_response,
                    standarized_response=standarized_response,
                    provider_job_id=provider_job_id,
                )
        elif job_status == "FAILED":
            #delete vocabulary
            try:
                clients["speech"].delete_vocabulary(
                    VocabularyName = vocab[0]
                )
            except IndexError as ir: # if not vocabulary was created
                pass
            except Exception as exc:
                raise ProviderException(str(exc)) from exc
            error = job_details["TranscriptionJob"].get("FailureReason")
            raise ProviderException(error)
        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )

    # Launch job label detection
    def video__label_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(provider_job_id=amazon_launch_video_job(file, "LABEL"))

    # Launch job text detection
    def video__text_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(provider_job_id=amazon_launch_video_job(file, "TEXT"))

    # Launch job face detection
    def video__face_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(provider_job_id=amazon_launch_video_job(file, "FACE"))

    # Launch job person tracking
    def video__person_tracking_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(provider_job_id=amazon_launch_video_job(file, "PERSON"))

    # Launch job explicit content detection
    def video__explicit_content_detection_async__launch_job(
        self, file: BufferedReader
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(provider_job_id=amazon_launch_video_job(file, "EXPLICIT"))

    # Get job result for label detection
    def video__label_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LabelDetectionAsyncDataClass]:
        pagination_token = ""
        max_result = 20
        finished = False
        while not finished:
            response = self.clients["video"].get_label_detection(
                JobId=provider_job_id,
                MaxResults=max_result,
                NextToken=pagination_token,
                SortBy="TIMESTAMP",
            )
            # jobstatus = response['JobStatus'] #SUCCEEDED, FAILED, IN_PROGRESS
            labels = []
            for label in response["Labels"]:
                # Category
                parents = []
                for parent in label["Label"]["Parents"]:
                    if parent["Name"]:
                        parents.append(parent["Name"])

                # bounding boxes
                boxes = []
                for instance in label["Label"]["Instances"]:
                    video_box = VideoLabelBoundingBox(
                        top=instance["BoundingBox"]["Top"],
                        left=instance["BoundingBox"]["Left"],
                        width=instance["BoundingBox"]["Width"],
                        height=instance["BoundingBox"]["Height"],
                    )
                    boxes.append(video_box)

                videolabel = VideoLabel(
                    timestamp=[
                        VideoLabelTimeStamp(start=float(label["Timestamp"]) / 1000.0)
                    ],
                    confidence=label["Label"]["Confidence"],
                    name=label["Label"]["Name"],
                    category=parents,
                    bounding_box=boxes,
                )
                labels.append(videolabel)

            standarized_response = LabelDetectionAsyncDataClass(labels=labels)
            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standarized_response, provider_job_id
        )

    # Get job result for text detection
    def video__text_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> TextDetectionAsyncDataClass:

        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = self.clients["video"].get_text_detection(
                JobId=provider_job_id,
                MaxResults=max_results,
                NextToken=pagination_token,
            )
            text_video = []
            # Get unique values of detected text annotation
            detected_texts = {
                text["TextDetection"]["DetectedText"]
                for text in response["TextDetections"]
            }

            # For each unique value, get all the frames where it appears
            for text in detected_texts:
                annotations = [
                    item
                    for item in response["TextDetections"]
                    if item["TextDetection"]["DetectedText"] == text
                ]
                frames = []
                for annotation in annotations:
                    timestamp = float(annotation["Timestamp"]) / 1000.0
                    confidence = annotation["TextDetection"]["Confidence"]
                    geometry = annotation["TextDetection"]["Geometry"]["BoundingBox"]
                    bounding_box = VideoTextBoundingBox(
                        top=geometry["Top"],
                        left=geometry["Left"],
                        width=geometry["Width"],
                        height=geometry["Height"],
                    )
                    frame = VideoTextFrames(
                        timestamp=timestamp,
                        confidence=confidence,
                        bounding_box=bounding_box,
                    )
                    frames.append(frame)

                video_text = VideoText(
                    text=text,
                    frames=frames,
                )
                text_video.append(video_text)

            standarized_response = TextDetectionAsyncDataClass(texts=text_video)

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standarized_response, provider_job_id
        )

    # Get job result for face detection
    def video__face_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> FaceDetectionAsyncDataClass:

        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = self.clients["video"].get_face_detection(
                JobId=provider_job_id,
                MaxResults=max_results,
                NextToken=pagination_token,
            )
            faces = []
            for face in response["Faces"]:
                # Time stamp
                offset = float(face["Timestamp"]) / 1000.0  # convert to seconds

                # Bounding box
                bounding_box = VideoBoundingBox(
                    top=face["Face"]["BoundingBox"]["Top"],
                    left=face["Face"]["BoundingBox"]["Left"],
                    height=face["Face"]["BoundingBox"]["Height"],
                    width=face["Face"]["BoundingBox"]["Width"],
                )

                # Attributes
                poses = VideoFacePoses(
                    pitch=face["Face"]["Pose"]["Pitch"] / 100,
                    yawn=face["Face"]["Pose"]["Yaw"] / 100,
                    roll=face["Face"]["Pose"]["Roll"] / 100,
                )
                attributes_video = FaceAttributes(
                    pose=poses,
                    brightness=face["Face"]["Quality"]["Brightness"] / 100,
                    sharpness=face["Face"]["Quality"]["Sharpness"] / 100,
                )

                # Landmarks
                landmarks_output = {}
                for land in face["Face"]["Landmarks"]:
                    if land.get("Type") and land.get("X") and land.get("Y"):
                        landmarks_output[land["Type"]] = [land["X"], land["Y"]]

                landmarks_video = LandmarksVideo(
                    eye_left=landmarks_output.get("eye_left", []),
                    eye_right=landmarks_output.get("eye_right", []),
                    mouth_left=landmarks_output.get("mouth_left", []),
                    mouth_right=landmarks_output.get("mouth_right", []),
                    nose=landmarks_output.get("nose", []),
                )
                faces.append(
                    VideoFace(
                        offset=offset,
                        attributes=attributes_video,
                        landmarks=landmarks_video,
                        bounding_box=bounding_box,
                    )
                )
            standarized_response = FaceDetectionAsyncDataClass(faces=faces)

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standarized_response, provider_job_id
        )

    # Get job result for person tracking
    def video__person_tracking_async__get_job_result(
        self, provider_job_id: str
    ) -> PersonTrackingAsyncDataClass:
        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = self.clients["video"].get_person_tracking(
                JobId=provider_job_id,
                MaxResults=max_results,
                NextToken=pagination_token,
            )

            # gather all persons with the same index :
            persons_index = {index["Person"]["Index"] for index in response["Persons"]}
            tracked_persons = []
            for index in persons_index:
                detected_persons = [
                    item
                    for item in response["Persons"]
                    if item["Person"]["Index"] == index
                ]
                tracked_person = []
                for detected_person in detected_persons:
                    if detected_person["Person"].get("BoundingBox"):
                        offset = float(detected_person["Timestamp"] / 1000.0)
                        bounding_box = detected_person.get("Person").get("BoundingBox")
                        bounding_box = VideoTrackingBoundingBox(
                            top=bounding_box["Top"],
                            left=bounding_box["Left"],
                            height=bounding_box["Height"],
                            width=bounding_box["Width"],
                        )
                        tracked_person.append(
                            PersonTracking(
                                offset=offset,
                                bounding_box=bounding_box,
                            )
                        )
                tracked_persons.append(VideoTrackingPerson(tracked=tracked_person))
            standarized_response = PersonTrackingAsyncDataClass(persons=tracked_persons)

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standarized_response, provider_job_id
        )

    # Get job result for explicit content detection
    def video__explicit_content_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> ExplicitContentDetectionAsyncDataClass:

        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = self.clients["video"].get_content_moderation(
                JobId=provider_job_id,
                MaxResults=max_results,
                NextToken=pagination_token,
            )
            moderated_content = []
            for label in response["ModerationLabels"]:
                confidence = label["ModerationLabel"]["Confidence"]
                timestamp = float(label["Timestamp"]) / 1000.0  # convert to seconds
                if label["ParentName"] != "":
                    category = label["ParentName"]
                else:
                    category = label["Name"]

                moderated_content.append(
                    ContentNSFW(
                        timestamp=timestamp, confidence=confidence, category=category
                    )
                )
            standarized_response = ExplicitContentDetectionAsyncDataClass(
                moderation=moderated_content
            )
            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standarized_response, provider_job_id
        )
