# pylint: disable=locally-disabled, too-many-lines
import base64
import io
import json
import os
from io import BufferedReader
from pathlib import Path
from time import time
from typing import Sequence

import googleapiclient.discovery
import numpy as np
from pdf2image.pdf2image import convert_from_bytes
from PIL import Image as Img
import google.auth
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import speech, storage, texttospeech
from google.cloud import translate_v3 as translate
from google.cloud import videointelligence, vision
from google.cloud.language import Document as GoogleDocument
from google.cloud.language import LanguageServiceClient
from google.cloud.vision_v1.types.image_annotator import (
    AnnotateImageResponse,
    EntityAnnotation,
)
from google.protobuf.json_format import MessageToDict
from edenai_apis.apis.google.google_helpers import (
    GoogleVideoFeatures,
    google_video_get_job,
    ocr_tables_async_response_add_rows,
    score_to_content,
    score_to_sentiment,
    get_tag_name,
)
from edenai_apis.features import Audio, Image, Ocr, ProviderApi, Text, Translation, Video
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceItem,
    FaceLandmarks,
    FacePoses,
    FaceQuality,
)
from edenai_apis.features.image.landmark_detection.landmark_detection_dataclass import (
    LandmarkDetectionDataClass,
    LandmarkItem,
    LandmarkLatLng,
    LandmarkLocation,
    LandmarkVertice,
)
from edenai_apis.features.image.logo_detection.logo_detection_dataclass import (
    LogoBoundingPoly,
    LogoDetectionDataClass,
    LogoItem,
    LogoVertice,
)
from edenai_apis.features.image.object_detection.object_detection_dataclass import (
    ObjectDetectionDataClass,
    ObjectItem,
)
from edenai_apis.features.ocr.ocr.ocr_dataclass import Bounding_box, OcrDataClass
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import (
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    Items,
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.syntax_analysis.syntax_analysis_dataclass import (
    InfosSyntaxAnalysisDataClass,
    SyntaxAnalysisDataClass,
)
from edenai_apis.features.translation.automatic_translation.automatic_translation_dataclass import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection.language_detection_dataclass import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.features.video import (
    ContentNSFW,
    ExplicitContentDetectionAsyncDataClass,
)
from edenai_apis.features.video.face_detection_async.face_detection_async_dataclass import (
    FaceAttributes,
    FaceDetectionAsyncDataClass,
    VideoBoundingBox,
    VideoFace,
)
from edenai_apis.features.video.label_detection_async.label_detection_async_dataclass import (
    LabelDetectionAsyncDataClass,
    VideoLabel,
    VideoLabelTimeStamp,
)
from edenai_apis.features.video.logo_detection_async.logo_detection_async_dataclass import (
    LogoDetectionAsyncDataClass,
    LogoTrack,
    VideoLogo,
    VideoLogoBoundingBox,
)
from edenai_apis.features.video.object_tracking_async.object_tracking_async_dataclass import (
    ObjectFrame,
    ObjectTrack,
    ObjectTrackingAsyncDataClass,
    VideoObjectBoundingBox,
)
from edenai_apis.features.video.person_tracking_async.person_tracking_async_dataclass import (
    LowerCloth,
    PersonAttributes,
    PersonLandmarks,
    PersonTracking,
    PersonTrackingAsyncDataClass,
    UpperCloth,
    VideoTrackingBoundingBox,
    VideoTrackingPerson,
)
from edenai_apis.features.video.text_detection_async.text_detection_async_dataclass import (
    TextDetectionAsyncDataClass,
    VideoText,
    VideoTextBoundingBox,
    VideoTextFrames,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.audio import wav_converter
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncErrorResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)



class GoogleApi(ProviderApi, Video, Audio, Image, Ocr, Text, Translation):
    provider_name = "google"

    def __init__(self):
        self.api_settings, location = load_provider(
            ProviderDataEnum.KEY, provider_name="google", location=True
        )
        self.project_id = self.api_settings["project_id"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = location

        self.clients = {
            "image": vision.ImageAnnotatorClient(),
            "text": LanguageServiceClient(),
            "storage": storage.Client(),
            "video": videointelligence.VideoIntelligenceServiceClient(),
            "translate": translate.TranslationServiceClient()
        }

    def image__object_detection(
        self, file: BufferedReader
    ) -> ResponseType[ObjectDetectionDataClass]:
        image = vision.Image(content=file.read())
        response = self.clients["image"].object_localization(image=image)
        response = MessageToDict(response._pb)
        items = []
        for object_annotation in response["localizedObjectAnnotations"]:
            x_min, x_max = np.infty, -np.infty
            y_min, y_max = np.infty, -np.infty
            # Getting borders
            for normalize_vertice in object_annotation["boundingPoly"][
                "normalizedVertices"
            ]:
                x_min, x_max = min(x_min, normalize_vertice["x"]), max(
                    x_max, normalize_vertice["x"]
                )
                y_min, y_max = min(y_min, normalize_vertice["y"]), max(
                    y_max, normalize_vertice["y"]
                )
                items.append(
                    ObjectItem(
                        label=object_annotation["name"],
                        confidence=object_annotation["score"],
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                    )
                )

        return ResponseType[ObjectDetectionDataClass](
            original_response=response,
            standarized_response=ObjectDetectionDataClass(items=items),
        )

    def image__face_detection(
        self, file: BufferedReader
    ) -> ResponseType[FaceDetectionDataClass]:
        file_content = file.read()
        img_size = Img.open(file).size
        image = vision.Image(content=file_content)
        response = self.clients["image"].face_detection(image=image, max_results=100)
        original_result = MessageToDict(response._pb)

        result = []
        width, height = img_size
        for face in original_result.get("faceAnnotations", []):

            # emotions
            emotions = FaceEmotions(
                joy=score_to_content(face.get("joyLikelihood")),
                sorrow=score_to_content(face.get("sorrowLikelihood")),
                anger=score_to_content(face.get("angerLikelihood")),
                surprise=score_to_content(face.get("surpriseLikelihood")),
            )

            # quality
            quality = FaceQuality(
                exposure=2
                * score_to_content(face.get("underExposedLikelihood", 0))
                / 10,
                blur=2 * score_to_content(face.get("blurredLikelihood", 0)) / 10,
            )

            # accessories
            headwear = 2 * score_to_content(face.get("headwearLikelihood", 0)) / 10
            accessories = FaceAccessories(headwear=headwear)

            # landmarks
            landmark_output = {}
            for land in face.get("landmarks", []):
                if "type" in land and "UNKNOWN_LANDMARK" not in land:
                    landmark_output[land["type"]] = [
                        land["position"]["x"] / width,
                        land["position"]["y"] / height,
                    ]
            landmarks = FaceLandmarks(
                left_eye=landmark_output.get("LEFT_EYE", []),
                left_eye_top=landmark_output.get("LEFT_EYE_TOP_BOUNDARY", []),
                left_eye_right=landmark_output.get("LEFT_EYE_RIGHT_CORNER", []),
                left_eye_bottom=landmark_output.get("LEFT_EYE_BOTTOM_BOUNDARY", []),
                left_eye_left=landmark_output.get("LEFT_EYE_LEFT_CORNER", []),
                right_eye=landmark_output.get("RIGHT_EYE", []),
                right_eye_top=landmark_output.get("RIGHT_EYE_TOP_BOUNDARY", []),
                right_eye_right=landmark_output.get("RIGHT_EYE_RIGHT_CORNER", []),
                right_eye_bottom=landmark_output.get("RIGHT_EYE_BOTTOM_BOUNDARY", []),
                right_eye_left=landmark_output.get("RIGHT_EYE_LEFT_CORNER", []),
                left_eyebrow_left=landmark_output.get("LEFT_OF_LEFT_EYEBROW", []),
                left_eyebrow_right=landmark_output.get("LEFT_OF_RIGHT_EYEBROW", []),
                left_eyebrow_top=landmark_output.get("LEFT_EYEBROW_UPPER_MIDPOINT", []),
                right_eyebrow_left=landmark_output.get("RIGHT_OF_LEFT_EYEBROW", []),
                right_eyebrow_right=landmark_output.get("RIGHT_OF_RIGHT_EYEBROW", []),
                nose_tip=landmark_output.get("NOSE_TIP", []),
                nose_bottom_right=landmark_output.get("NOSE_BOTTOM_RIGHT", []),
                nose_bottom_left=landmark_output.get("NOSE_BOTTOM_LEFT", []),
                mouth_left=landmark_output.get("MOUTH_LEFT", []),
                mouth_right=landmark_output.get("MOUTH_RIGHT", []),
                right_eyebrow_top=landmark_output.get(
                    "RIGHT_EYEBROW_UPPER_MIDPOINT", []
                ),
                midpoint_between_eyes=landmark_output.get("MIDPOINT_BETWEEN_EYES", []),
                nose_bottom_center=landmark_output.get("NOSE_BOTTOM_CENTER", []),
                upper_lip=landmark_output.get("GET_UPPER_LIP", []),
                under_lip=landmark_output.get("GET_LOWER_LIP", []),
                mouth_center=landmark_output.get("MOUTH_CENTER", []),
                left_ear_tragion=landmark_output.get("LEFT_EAR_TRAGION", []),
                right_ear_tragion=landmark_output.get("RIGHT_EAR_TRAGION", []),
                forehead_glabella=landmark_output.get("FOREHEAD_GLABELLA", []),
                chin_gnathion=landmark_output.get("CHIN_GNATHION", []),
                chin_left_gonion=landmark_output.get("CHIN_LEFT_GONION", []),
                chin_right_gonion=landmark_output.get("CHIN_RIGHT_GONION", []),
                left_cheek_center=landmark_output.get("LEFT_CHEEK_CENTER", []),
                right_cheek_center=landmark_output.get("RIGHT_CHEEK_CENTER", []),
            )

            # bounding box
            bounding_poly = face.get("fdBoundingPoly", {}).get("vertices", [])

            result.append(
                FaceItem(
                    accessories=accessories,
                    quality=quality,
                    emotions=emotions,
                    landmarks=landmarks,
                    poses=FacePoses(
                        roll=face.get("rollAngle"),
                        pitch=face.get("panAngle"),
                        yaw=face.get("tiltAngle"),
                    ),
                    confidence=face.get("detectionConfidence"),
                    # indices are this way because array of bounding boxes
                    # follow this pattern:
                    # [top-left, top-right, bottom-right, bottom-left]
                    bounding_box=FaceBoundingBox(
                        x_min=bounding_poly[0].get("x", 0.0) / width,
                        x_max=bounding_poly[1].get("x", width) / width,
                        y_min=bounding_poly[0].get("y", 0.0) / height,
                        y_max=bounding_poly[3].get("y", height) / height,
                    ),
                )
            )
        return ResponseType[FaceDetectionDataClass](
            original_response=original_result,
            standarized_response=FaceDetectionDataClass(items=result),
        )

    def image__landmark_detection(
        self, file: BufferedReader
    ) -> ResponseType[LandmarkDetectionDataClass]:
        image = vision.Image(content=file.read())
        response = self.clients["image"].landmark_detection(image=image)
        dict_response = vision.AnnotateImageResponse.to_dict(response)
        landmarks = dict_response.get("landmark_annotations", [])

        items: Sequence[LandmarkItem] = []
        for landmark in landmarks:
            if landmark.get("description") not in [item.description for item in items]:
                vertices: Sequence[LandmarkVertice] = []
                for poly in landmark.get("bounding_poly", {}).get("vertices", []):
                    vertices.append(LandmarkVertice(x=poly["x"], y=poly["y"]))
                locations = []
                for location in landmark.get("locations", []):
                    locations.append(
                        LandmarkLocation(
                            lat_lng=LandmarkLatLng(
                                latitude=location.get("lat_lng", {}).get("latitude"),
                                longitude=location.get("lat_lng", {}).get("longitude"),
                            )
                        )
                    )
                items.append(
                    LandmarkItem(
                        description=landmark.get("description"),
                        confidence=landmark.get("score"),
                        bounding_box=vertices,
                        locations=locations,
                    )
                )
        if dict_response.get("error"):
            raise ProviderException(
                message=dict_response["error"].get(
                    "message", "Error calling Google Api"
                )
            )

        return ResponseType[LandmarkDetectionDataClass](
            original_response=landmarks,
            standarized_response=LandmarkDetectionDataClass(items=items),
        )

    def image__logo_detection(
        self, file: BufferedReader
    ) -> ResponseType[LogoDetectionDataClass]:
        image = vision.Image(content=file.read())
        response = self.clients["image"].logo_detection(image=image)
        response = MessageToDict(response._pb)

        # Handle error
        if response.get("error", {}).get("message"):
            raise Exception(
                f"{response.get('error', {}).get('message')}\n"
                + "For more info on error messages, check: "
                + "https://cloud.google.com/apis/design/errors"
            )

        items: Sequence[LogoItem] = []
        for key in response.get("logoAnnotations", []):
            vertices = []
            for vertice in key.get("boundingPoly").get("vertices"):
                vertices.append(
                    LogoVertice(x=float(vertice.get("x")), y=float(vertice.get("y")))
                )

            items.append(
                LogoItem(
                    description=key.get("description"),
                    score=key.get("score"),
                    bounding_poly=LogoBoundingPoly(vertices=vertices),
                )
            )
        return ResponseType[LogoDetectionDataClass](
            original_response=response,
            standarized_response=LogoDetectionDataClass(items=items),
        )

    def audio__text_to_speech(
        self, language: str, text: str, option: str
    ) -> ResponseType[TextToSpeechDataClass]:
        voice_type = 1
        ssml_gender = None

        if language in ["da-DK", "pt-BR", "es-ES"] and option == "MALE":
            option = "FEMALE"
            voice_type = 0

        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)

        if option == "FEMALE":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
        else:
            ssml_gender = texttospeech.SsmlVoiceGender.MALE

        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            ssml_gender=ssml_gender,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Getting response of API
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        audio = base64.b64encode(response.audio_content).decode("utf-8")

        standarized_response = TextToSpeechDataClass(audio=audio, voice_type=voice_type)
        return ResponseType[TextToSpeechDataClass](
            original_response={},
            standarized_response=standarized_response,
        )

    def ocr__ocr_tables_async__launch_job(
        self, file: BufferedReader, file_type: str, language: str
    ) -> AsyncLaunchJobResponseType:
        file_name: str = file.name.split("/")[-1]  # file.name give its whole path

        documentai_projectid = self.api_settings["documentai"]["project_id"]
        documentai_processid = self.api_settings["documentai"]["process_id"]

        gcs_output_uri = "gs://async-ocr-tables"
        gcs_output_uri_prefix = "outputs"
        gcs_input_uri = f"gs://async-ocr-tables/{file_name}"

        # upload file to bucket
        bucket_client = self.clients["storage"]
        ocr_tables_bucket = bucket_client.get_bucket("async-ocr-tables")
        new_blob = ocr_tables_bucket.blob(file_name)
        new_blob.upload_from_string(file.read())

        doc_ai = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": "eu-documentai.googleapis.com"}
        )

        destination_uri = f"{gcs_output_uri}/{gcs_output_uri_prefix}/"

        gcs_documents = documentai.GcsDocuments(
            documents=[{"gcs_uri": gcs_input_uri, "mime_type": file_type}]
        )

        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config={"gcs_uri": destination_uri}
        )

        name = f"projects/{documentai_projectid}/locations/eu/processors/{documentai_processid}"

        request = documentai.BatchProcessRequest(
            name=name,
            input_documents=input_config,
            document_output_config=output_config,
        )
        response = doc_ai.batch_process_documents(request)

        operation_id = response.operation.name.split("/")[-1]

        return AsyncLaunchJobResponseType(provider_job_id=operation_id)

    def ocr__ocr_tables_async__get_job_result(
        self, job_id: str
    ) -> ResponseType[OcrTablesAsyncDataClass]:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials, _ = google.auth.default(scopes=scopes)

        documentai_projectid = self.api_settings["documentai"]["project_id"]

        name = f"projects/{documentai_projectid}/locations/eu/operations/{job_id}"

        service = googleapiclient.discovery.build(
            serviceName="documentai",
            version="v1beta3",
            credentials=credentials,
            client_options={"api_endpoint": "https://eu-documentai.googleapis.com"},
        )

        request = service.projects().locations().operations().get(name=name)
        res = request.execute()

        if res["metadata"]["state"] == "SUCCEEDED":
            # get result from bucket
            bucket_client = self.clients["storage"]
            ocr_tables_bucket = bucket_client.get_bucket("async-ocr-tables")
            output_uri = res["metadata"]["individualProcessStatuses"][0][
                "outputGcsDestination"
            ]
            prefix = output_uri.split("gs://async-ocr-tables/")[1] + "/"
            blob_list = ocr_tables_bucket.list_blobs(prefix=prefix)
            # convert byte array to json
            byte_res = list(blob_list)[0].download_as_string()
            original_result = json.loads(byte_res.decode("utf8"))

            raw_text = original_result["text"]
            pages: Sequence[Page] = []
            num_pages = len(original_result["pages"])
            for page in original_result["pages"]:
                tables: Sequence[Table] = []
                if "tables" in page.keys():
                    for table in page["tables"]:
                        ocr_num_rows = 0
                        ocr_num_cols = 0
                        rows: Sequence[Row] = []
                        if "headerRows" in table.keys():
                            for row in table["headerRows"]:
                                ocr_num_rows += 1
                                row, num_row_cols = ocr_tables_async_response_add_rows(
                                    row, raw_text, is_header=True
                                )
                                ocr_num_cols = max(ocr_num_cols, num_row_cols)
                                rows.append(row)
                        if "bodyRows" in table.keys():
                            for row in table["bodyRows"]:
                                ocr_num_rows += 1
                                row, num_row_cols = ocr_tables_async_response_add_rows(
                                    row, raw_text
                                )
                                ocr_num_cols = max(ocr_num_cols, num_row_cols)
                                rows.append(row)
                        ocr_table = Table(
                            rows=rows, num_rows=ocr_num_rows, num_cols=ocr_num_cols
                        )
                        tables.append(ocr_table)
                    ocr_page = Page(tables=tables)
                    pages.append(ocr_page)
            standarized_response = OcrTablesAsyncDataClass(
                pages=pages, num_pages=num_pages
            )

            return AsyncResponseType[OcrTablesAsyncDataClass](
                status="succeeded",
                original_response=original_result,
                standarized_response=standarized_response,
                provider_job_id=job_id,
            )

        elif res["metadata"]["state"] == "FAILED":
            return AsyncErrorResponseType[OcrTablesAsyncDataClass](
                status="failed",
                error=res.get("error"),
                provider_job_id=job_id,
            )
        return AsyncPendingResponseType[OcrTablesAsyncDataClass](
            status="pending", provider_job_id=job_id
        )

    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        """

        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )
        # Getting response of API
        response = self.clients["text"].analyze_entities(
            document=document,
            encoding_type="UTF8",
        )

        # Create output response
        # Convert response to dict
        response = MessageToDict(response._pb)
        items: Sequence[InfosNamedEntityRecognitionDataClass] = []

        # Analyse response
        # Getting name of entity, its category and its score of confidence
        if response.get("entities") and isinstance(response["entities"], list):
            for ent in response["entities"]:
                items.append(
                    InfosNamedEntityRecognitionDataClass(
                        entity=ent["name"],
                        importance=ent.get("salience", None),
                        category=ent["type"],
                        url=ent.get("metadata", {}).get("wikipedia_url", None),
                    )
                )

        standarized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=response, standarized_response=standarized_response
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        :return:                Array that contain api response and TextSentimentAnalysis
        Object that contains the sentiments and their rates
        """

        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )

        # Getting response of API
        response = self.clients["text"].analyze_sentiment(
            document=document,
            encoding_type="UTF8",
        )

        # Convert response to dict
        response = MessageToDict(response._pb)
        # Create output response
        items: Sequence[Items] = []
        items.append(
            Items(
                sentiment=score_to_sentiment(
                    response["documentSentiment"].get("score", 0)
                ),
                sentiment_rate=abs(response["documentSentiment"].get("score", 0)),
            )
        )
        standarize = SentimentAnalysisDataClass(items=items)

        return ResponseType[SentimentAnalysisDataClass](
            original_response=response, standarized_response=standarize
        )

    def text__syntax_analysis(
        self, language: str, text: str
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        :return:                Array containing api response and TextSyntaxAnalysis Object
        that contains the sentiments and their syntax
        """

        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )
        # Getting response of API
        response = self.clients["text"].analyze_syntax(
            document=document,
            encoding_type="UTF8",
        )
        # Convert response to dict
        response = MessageToDict(response._pb)

        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        # Analysing response
        # Getting syntax detected of word and its score of confidence
        for token in response["tokens"]:

            part_of_speech_tag = {}
            part_of_speech_filter = {}
            part_of_speech = token["partOfSpeech"]
            part_of_speech_keys = list(part_of_speech.keys())
            part_of_speech_values = list(part_of_speech.values())
            for key, prop in enumerate(part_of_speech_keys):
                tag_ = ""
                if "proper" in part_of_speech_keys[key]:
                    prop = "proper_name"
                if "UNKNOWN" not in part_of_speech_values[key]:
                    if "tag" in prop:
                        tag_ = get_tag_name(part_of_speech_values[key])
                        part_of_speech_tag[prop] = tag_
                    else:
                        part_of_speech_filter[prop] = part_of_speech_values[key]

            items.append(
                InfosSyntaxAnalysisDataClass(
                    word=token["text"]["content"],
                    tag=part_of_speech_tag["tag"],
                    lemma=token["lemma"],
                    others=part_of_speech_filter,
                )
            )

        standarized_response = SyntaxAnalysisDataClass(items=items)

        result = ResponseType[SyntaxAnalysisDataClass](
            original_response=response,
            standarized_response=standarized_response,
        )
        return result

    def audio__speech_to_text_async__launch_job(
        self, file: BufferedReader, language: str, speakers: int
    ) -> AsyncLaunchJobResponseType:
        export_format = "flac"
        wav_file, _, _, channels = wav_converter(file, export_format)
        audio_name = str(int(time())) + Path(file.name).stem + "." + export_format
        # Upload file to google cloud
        storage_client: storage.Client = self.clients["storage"]
        bucket_name = "audios-speech2text"
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(audio_name)
        blob.upload_from_file(wav_file)

        # blob.download_to_filename(audio_name)

        gcs_uri = f"gs://{bucket_name}/{audio_name}"
        # Launch file transcription
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        diarization = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=speakers,
        )
        config = speech.RecognitionConfig(
            # encoding="LINEAR16",
            language_code=language,
            audio_channel_count=channels,
            diarization_config = diarization
            # sample_rate_hertz=frame_rate
        )
        operation = client.long_running_recognize(config=config, audio=audio)
        operation_name = operation.operation.name
        return AsyncLaunchJobResponseType(provider_job_id=operation_name)

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        service = googleapiclient.discovery.build("speech", "v1")
        service_request_ = service.operations().get(name=provider_job_id)
        original_response = service_request_.execute()

        if original_response.get("error") is not None:
            return AsyncErrorResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        text = ""
        diarization = SpeechDiarization(total_speakers=0, entries= [])
        if original_response.get("done"):
            if original_response["response"].get("results"):
                text = ", ".join(
                    [
                        entry["alternatives"][0]["transcript"].strip() if 
                        entry["alternatives"][0].get("transcript") else ""
                        for entry in original_response["response"]["results"]
                    ]
                )

                diarization_entries = []
                result = original_response["response"]["results"][-1]
                words_info = result["alternatives"][0]["words"]
                speakers = set()

                for word_info in words_info:
                    speakers.add(word_info['speakerTag'])
                    diarization_entries.append(
                        SpeechDiarizationEntry(
                            segment= word_info['word'],
                            speaker= word_info['speakerTag'],
                            start_time= word_info['startTime'][:-1],
                            end_time= word_info['endTime'][:-1]
                        )
                    )
                
                diarization = SpeechDiarization(total_speakers=len(speakers), entries= diarization_entries)
            
            standarized_response = SpeechToTextAsyncDataClass(text=text, diarization=diarization)
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )
        return AsyncPendingResponseType(provider_job_id=provider_job_id)

    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:

        image = vision.Image(content=file.read())
        response = self.clients["image"].safe_search_detection(image=image)
        # Convert response to dict
        response = MessageToDict(response._pb)

        # Analyse response
        # Getting the explicit label and its score of image
        response = response["safeSearchAnnotation"]

        items = [
            ExplicitItem(label="Adult", likelihood=score_to_content(response["adult"])),
            ExplicitItem(label="Spoof", likelihood=score_to_content(response["spoof"])),
            ExplicitItem(
                label="Medical", likelihood=score_to_content(response["medical"])
            ),
            ExplicitItem(
                label="Gore", likelihood=score_to_content(response["violence"])
            ),
            ExplicitItem(label="Racy", likelihood=score_to_content(response["racy"])),
        ]

        return ResponseType(
            original_response=response,
            standarized_response=ExplicitContentDataClass(items=items),
        )

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        # Getting response
        client = self.clients['translate']
        parent = f"projects/{self.project_id}/locations/global"

        response = client.translate_text(
            parent=parent,
            contents=[text],
            mime_type="text/plain",  # mime types: text/plain, text/html
            source_language_code=source_language,
            target_language_code=target_language,
        )

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
            original_response=MessageToDict(response._pb), standarized_response=std
        )

    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        client = self.clients['translate']
        parent = f"projects/{self.project_id}/locations/global"
        response = client.detect_language(
            parent=parent,
            content=text,
            mime_type="text/plain",
        )

        items: Sequence[InfosLanguageDetectionDataClass] = []
        for language in response.languages:
            items.append(
                InfosLanguageDetectionDataClass(
                    language=language.language_code, confidence=language.confidence
                )
            )

        standarized_response = LanguageDetectionDataClass(items=items)

        return ResponseType[LanguageDetectionDataClass](
            original_response=MessageToDict(response._pb),
            standarized_response=standarized_response,
        )

    def ocr__ocr(
        self, file: BufferedReader, language: str
    ) -> ResponseType[OcrDataClass]:
        is_pdf = file.name.lower().endswith(".pdf")
        responses = []
        index = 1
        file_content = file.read()

        if is_pdf:
            ocr_file_images = convert_from_bytes(
                file_content, fmt="jpeg", poppler_path=None
            )
            for ocr_image in ocr_file_images:
                ocr_image_buffer = io.BytesIO()
                ocr_image.save(ocr_image_buffer, format="JPEG")
                image = vision.Image(content=ocr_image_buffer.getvalue())
                response = self.clients["image"].text_detection(image=image)
                responses.append((response, ocr_image.size))
                index += 1
        else:
            image = vision.Image(content=file_content)
            response = self.clients["image"].text_detection(image=image)
            responses.append((response, Img.open(file).size))

        messages_list = []
        boxes: Sequence[Bounding_box] = []
        final_text = ""
        for output in responses:
            image_response: AnnotateImageResponse = output[0]
            # TO DO better original_response
            messages_list.append(image_response)

            # Get width and hight
            width, hight = output[1]

            text_annotations: Sequence[
                EntityAnnotation
            ] = image_response.text_annotations
            final_text += text_annotations[0].description.replace("\n", " ")
            for text in text_annotations[1:]:
                xleft = float(text.bounding_poly.vertices[0].x)
                xright = float(text.bounding_poly.vertices[1].x)
                ytop = float(text.bounding_poly.vertices[0].y)
                ybottom = float(text.bounding_poly.vertices[2].y)
                boxes.append(
                    Bounding_box(
                        text=text.description,
                        left=float(xleft / width),
                        top=float(ytop / hight),
                        width=(xright - xleft) / width,
                        height=(ybottom - ytop) / hight,
                    )
                )
        standarized = OcrDataClass(
            text=final_text.replace("\n", " ").strip(), bounding_boxes=boxes
        )
        return ResponseType[OcrDataClass](
            original_response=messages_list, standarized_response=standarized
        )

    def google_video_launch_job(
        self, file: BufferedReader, feature: GoogleVideoFeatures
    ) -> AsyncLaunchJobResponseType:
        # Launch async job for label detection
        storage_client = self.clients["storage"]
        bucket_name = "audios-speech2text"
        file_extension = file.name.split(".")[-1]
        file_name = (
            str(int(time())) + Path(file.name).stem + "_video_." + file_extension
        )

        # Upload video to GCS
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)

        blob.upload_from_string(file.read())
        gcs_uri = f"gs://{bucket_name}/{file_name}"

        # Configure the request for each feature
        features = {
            GoogleVideoFeatures.LABEL: self.clients["video"].annotate_video(
                request={
                    "features": [videointelligence.Feature.LABEL_DETECTION],
                    "input_uri": gcs_uri,
                }
            ),
            GoogleVideoFeatures.TEXT: self.clients["video"].annotate_video(
                request={
                    "features": [videointelligence.Feature.TEXT_DETECTION],
                    "input_uri": gcs_uri,
                }
            ),
            GoogleVideoFeatures.FACE: self.clients["video"].annotate_video(
                request={
                    "features": [videointelligence.Feature.FACE_DETECTION],
                    "input_uri": gcs_uri,
                    "video_context": videointelligence.VideoContext(
                        face_detection_config=videointelligence.FaceDetectionConfig(
                            include_bounding_boxes=True, include_attributes=True
                        )
                    ),
                }
            ),
            GoogleVideoFeatures.PERSON: self.clients["video"].annotate_video(
                request={
                    "features": [videointelligence.Feature.PERSON_DETECTION],
                    "input_uri": gcs_uri,
                    "video_context": videointelligence.VideoContext(
                        person_detection_config=videointelligence.PersonDetectionConfig(
                            include_bounding_boxes=True,
                            include_attributes=True,
                            include_pose_landmarks=True,
                        )
                    ),
                }
            ),
            GoogleVideoFeatures.LOGO: self.clients["video"].annotate_video(
                request={
                    "features": [videointelligence.Feature.LOGO_RECOGNITION],
                    "input_uri": gcs_uri,
                }
            ),
            GoogleVideoFeatures.OBJECT: self.clients["video"].annotate_video(
                request={
                    "features": [videointelligence.Feature.OBJECT_TRACKING],
                    "input_uri": gcs_uri,
                }
            ),
            GoogleVideoFeatures.EXPLICIT: self.clients["video"].annotate_video(
                request={
                    "features": [videointelligence.Feature.EXPLICIT_CONTENT_DETECTION],
                    "input_uri": gcs_uri,
                }
            ),
        }

        # Return job id (operation name)
        return AsyncLaunchJobResponseType(provider_job_id=features[feature].operation.name)

    # Launch label detection job
    def video__label_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.LABEL)

    # Launch text detection job
    def video__text_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.TEXT)

    # Launch face detection job
    def video__face_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.FACE)

    # Launch person tracking job
    def video__person_tracking_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.PERSON)

    # Launch logo detection job
    def video__logo_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.LOGO)

    # Launch object tracking job
    def video__object_tracking_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.OBJECT)

    # Launch explicit content detection job
    def video__explicit_content_detection_async__launch_job(
        self, file: BufferedReader
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.EXPLICIT)

    def video__label_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LabelDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)
        if result.get("done"):
            annotations = result["response"]["annotationResults"][0]
            label = (
                annotations["segmentLabelAnnotations"]
                + annotations["shotLabelAnnotations"]
            )
            label_list = []

            for entity in label:
                confidences = []
                timestamps = []
                categories = []
                name = entity["entity"]["description"]
                for segment in entity["segments"]:
                    confidences.append(segment["confidence"])
                    start = segment["segment"]["startTimeOffset"][:-1]
                    end = segment["segment"]["endTimeOffset"][:-1]
                    timestamps.append(
                        VideoLabelTimeStamp(start=float(start), end=float(end))
                    )
                if entity.get("categoryEntities"):
                    for category in entity["categoryEntities"]:
                        categories.append(category["description"])

                label_list.append(
                    VideoLabel(
                        name=name,
                        category=categories,
                        confidence=(sum(confidences) / len(confidences)) * 100,
                        timestamp=timestamps,
                    )
                )
            standarized_response = LabelDetectionAsyncDataClass(labels=label_list)

            return AsyncResponseType[LabelDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType(
            status="pending", provider_job_id=provider_job_id
        )

    def video__text_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[TextDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)
        if result.get("done"):
            annotations = result["response"]["annotationResults"][0]
            texts = []
            for annotation in annotations["textAnnotations"]:
                frames = []
                description = annotation["text"]
                for segment in annotation["segments"]:
                    confidence = segment["confidence"]
                    for frame in segment["frames"]:
                        offset = frame["timeOffset"]
                        timestamp = float(offset[:-1])
                        xleft = frame["rotatedBoundingBox"]["vertices"][0]["x"]
                        xright = frame["rotatedBoundingBox"]["vertices"][1]["x"]
                        ytop = frame["rotatedBoundingBox"]["vertices"][0]["y"]
                        ybottom = frame["rotatedBoundingBox"]["vertices"][2]["y"]
                        bounding_box = VideoTextBoundingBox(
                            top=ytop,
                            left=xleft,
                            width=(xright - xleft),
                            height=(ybottom - ytop),
                        )
                        frames.append(
                            VideoTextFrames(
                                confidence=float(confidence),
                                timestamp=timestamp,
                                bounding_box=bounding_box,
                            )
                        )
                texts.append(VideoText(text=description, frames=frames))
            standarized_response = TextDetectionAsyncDataClass(texts=texts)
            return AsyncResponseType[TextDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType(
            status="pending", provider_job_id=provider_job_id
        )

    def video__face_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[FaceDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)
        if result.get("done"):
            faces = []
            response = result["response"]["annotationResults"][0]
            if response.get("faceDetectionAnnotations") is not None:
                for annotation in response['faceDetectionAnnotations']:
                    for track in annotation["tracks"]:
                        timestamp = float(track["timestampedObjects"][0]["timeOffset"][:-1])
                        bounding_box = VideoBoundingBox(
                            top=track["timestampedObjects"][0]["normalizedBoundingBox"][
                                "top"
                            ],
                            left=track["timestampedObjects"][0]["normalizedBoundingBox"][
                                "left"
                            ],
                            height=track["timestampedObjects"][0]["normalizedBoundingBox"][
                                "bottom"
                            ],
                            width=track["timestampedObjects"][0]["normalizedBoundingBox"][
                                "right"
                            ],
                        )
                        attribute_dict = {}
                        for attr in track["timestampedObjects"][0].get("attributes", []):
                            attribute_dict[attr["name"]] = attr["confidence"]
                        attributs = FaceAttributes(
                            headwear=attribute_dict["headwear"],
                            frontal_gaze=attribute_dict["headwear"],
                            eyes_visible=attribute_dict["eyes_visible"],
                            glasses=attribute_dict["glasses"],
                            mouth_open=attribute_dict["mouth_open"],
                            smiling=attribute_dict["smiling"],
                        )
                        face = VideoFace(
                            offset=timestamp,
                            bounding_box=bounding_box,
                            attributes=attributs,
                        )
                        faces.append(face)
            standarized_response = FaceDetectionAsyncDataClass(faces=faces)
            return AsyncResponseType[FaceDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[FaceDetectionAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def video__person_tracking_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[PersonTrackingAsyncDataClass]:
        result = google_video_get_job(provider_job_id)
        if result.get("done"):
            response = result["response"]["annotationResults"][0]
            persons = response["personDetectionAnnotations"]
            tracked_persons = []
            for person in persons:
                tracked_person = []
                for track in person["tracks"]:
                    for time_stamped_object in track["timestampedObjects"]:

                        # Bounding box
                        bounding_box = VideoTrackingBoundingBox(
                            top=float(
                                time_stamped_object["normalizedBoundingBox"]["top"]
                            ),
                            left=float(
                                time_stamped_object["normalizedBoundingBox"]["left"]
                            ),
                            height=float(
                                time_stamped_object["normalizedBoundingBox"]["bottom"]
                            ),
                            width=float(
                                time_stamped_object["normalizedBoundingBox"]["right"]
                            ),
                        )

                        # Timeoffset
                        timeoffset = float(time_stamped_object["timeOffset"][:-1])

                        # attributes
                        upper_clothes = []
                        lower_clothes = []
                        for attr in time_stamped_object.get("attributes", []):
                            if "Upper" in attr["name"]:
                                upper_clothes.append(
                                    UpperCloth(
                                        value=attr["value"],
                                        confidence=attr["confidence"],
                                    )
                                )
                            if "Lower" in attr["name"]:
                                lower_clothes.append(
                                    LowerCloth(
                                        value=attr["value"],
                                        confidence=attr["confidence"],
                                    )
                                )
                        tracked_attributes = PersonAttributes(
                            upper_cloths=upper_clothes, lower_cloths=lower_clothes
                        )

                        # Landmarks
                        landmark_output = {}
                        for land in time_stamped_object.get("landmarks", []):
                            landmark_output[land["name"]] = [
                                land["point"]["x"],
                                land["point"]["y"],
                            ]
                        landmark_tracking = PersonLandmarks(
                            nose=landmark_output.get("nose", []),
                            eye_left=landmark_output.get("left_eye", []),
                            eye_right=landmark_output.get("right_eye", []),
                            shoulder_left=landmark_output.get("left_shoulder", []),
                            shoulder_right=landmark_output.get("right_shoulder", []),
                            elbow_left=landmark_output.get("left_elbow", []),
                            elbow_right=landmark_output.get("right_elbow", []),
                            wrist_left=landmark_output.get("left_wrist", []),
                            wrist_right=landmark_output.get("right_wrist", []),
                            hip_left=landmark_output.get("left_hip", []),
                            hip_right=landmark_output.get("right_hip", []),
                            knee_left=landmark_output.get("left_knee", []),
                            knee_right=landmark_output.get("right_knee", []),
                            ankle_left=landmark_output.get("left_ankle", []),
                            ankle_right=landmark_output.get("right_ankle", []),
                        )

                        # Create tracked person
                        tracked_person.append(
                            PersonTracking(
                                offset=timeoffset,
                                attributes=tracked_attributes,
                                landmarks=landmark_tracking,
                                bounding_box=bounding_box,
                            )
                        )
                tracked_persons.append(VideoTrackingPerson(tracked=tracked_person))
            standarized_response = PersonTrackingAsyncDataClass(persons=tracked_persons)

            return AsyncResponseType[PersonTrackingAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[PersonTrackingAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def video__logo_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LogoDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)
        print('RESULT', result)
        if result.get("done"):
            response = result["response"]["annotationResults"][0]
            tracks = []
            print('RESPONSE', response)
            if 'logoRecognitionAnnotations' in response:
                for logo in response["logoRecognitionAnnotations"]:
                    objects = []
                    description = logo["entity"]["description"]
                    for track in logo["tracks"]:
                        for time_stamped_object in track["timestampedObjects"]:
                            timestamp = float(time_stamped_object["timeOffset"][:-1])
                            bounding_box = VideoLogoBoundingBox(
                                top=time_stamped_object["normalizedBoundingBox"]["top"],
                                left=time_stamped_object["normalizedBoundingBox"]["left"],
                                height=time_stamped_object["normalizedBoundingBox"][
                                    "bottom"
                                ],
                                width=time_stamped_object["normalizedBoundingBox"]["right"],
                            )
                            objects.append(
                                VideoLogo(
                                    timestamp=timestamp,
                                    bounding_box=bounding_box,
                                )
                            )
                    tracks.append(LogoTrack(description=description, tracking=objects))
            standarized_response = LogoDetectionAsyncDataClass(logos=tracks)

            return AsyncResponseType[LogoDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[LogoDetectionAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def video__object_tracking_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[ObjectTrackingAsyncDataClass]:
        result = google_video_get_job(provider_job_id)
        if result.get("done"):
            response = result["response"]["annotationResults"][0]
            objects = response["objectAnnotations"]
            object_tracking = []
            for detected_object in objects:
                frames = []
                description = detected_object["entity"]["description"]
                for frame in detected_object["frames"]:
                    timestamp = float(frame["timeOffset"][:-1])
                    bounding_box = VideoObjectBoundingBox(
                        top=float(frame["normalizedBoundingBox"]["top"]),
                        left=float(frame["normalizedBoundingBox"]["left"]),
                        width=float(frame["normalizedBoundingBox"]["right"]),
                        height=float(frame["normalizedBoundingBox"]["bottom"]),
                    )
                    frames.append(
                        ObjectFrame(timestamp=timestamp, bounding_box=bounding_box)
                    )
                object_tracking.append(
                    ObjectTrack(description=description, frames=frames)
                )
            standarized_response = ObjectTrackingAsyncDataClass(objects=object_tracking)
            return AsyncResponseType[ObjectTrackingAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[ObjectTrackingAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def video__explicit_content_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[ExplicitContentDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)
        if result.get("done"):
            response = result["response"]["annotationResults"][0]
            moderation = response["explicitAnnotation"]["frames"]
            label_list = []
            for label in moderation:
                timestamp = float(label["timeOffset"][:-1])
                category = "Explicit Nudity"
                confidence = float(score_to_content(label["pornographyLikelihood"]) / 5)
                label_list.append(
                    ContentNSFW(
                        timestamp=timestamp, category=category, confidence=confidence
                    )
                )
            standarized_response = ExplicitContentDetectionAsyncDataClass(
                moderation=label_list
            )
            return AsyncResponseType[ExplicitContentDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[ExplicitContentDetectionAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )
