from pathlib import Path
from time import time, sleep
from typing import List, Dict, Any
import requests
import json

from google.cloud import videointelligence

from edenai_apis.apis.google.google_helpers import (
    GoogleVideoFeatures,
    google_video_get_job,
    score_to_content,
    calculate_usage_tokens,
)
from edenai_apis.features.video import (
    ContentNSFW,
    ExplicitContentDetectionAsyncDataClass,
    QuestionAnswerDataClass,
    QuestionAnswerAsyncDataClass,
)
from edenai_apis.features.video.face_detection_async.face_detection_async_dataclass import (
    FaceAttributes,
    FaceDetectionAsyncDataClass,
    LandmarksVideo,
    VideoBoundingBox,
    VideoFace,
    VideoFacePoses,
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
    VideoPersonPoses,
    VideoPersonQuality,
    VideoTrackingBoundingBox,
    VideoTrackingPerson,
)
from edenai_apis.features.video.text_detection_async.text_detection_async_dataclass import (
    TextDetectionAsyncDataClass,
    VideoText,
    VideoTextBoundingBox,
    VideoTextFrames,
)
from edenai_apis.features.video.shot_change_detection_async.shot_change_detection_async_dataclass import (
    ShotChangeDetectionAsyncDataClass,
    ShotFrame,
)
from edenai_apis.apis.amazon.helpers import check_webhook_result
from edenai_apis.features.video.video_interface import VideoInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)


class GoogleVideoApi(VideoInterface):
    def google_upload_video(
        self,
        file: str,
    ) -> AsyncLaunchJobResponseType:
        # Launch async job for label detection
        storage_client = self.clients["storage"]
        bucket_name = "audios-speech2text"
        file_extension = file.split(".")[-1]
        file_name = str(int(time())) + Path(file).stem + "_video_." + file_extension

        # Upload video to GCS
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)

        blob.upload_from_filename(file)
        gcs_uri = f"gs://{bucket_name}/{file_name}"

        return gcs_uri

    def _check_file_status(self, file_uri: str, api_key: str) -> Dict[str, Any]:
        url = f"{file_uri}?key={api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        try:
            response_json = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        return response_json

    def _upload_and_process_file(self, file: str, api_key: str) -> Dict[str, Any]:
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"

        with open(file, "rb") as video_file:
            file = {"file": video_file}
            response = requests.post(upload_url, files=file)

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        try:
            file_data = response.json()["file"]
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        return file_data

    def _delete_file(self, file: str, api_key: str):
        delete_url = (
            f"https://generativelanguage.googleapis.com/v1beta/{file}?key={api_key}"
        )
        response = requests.delete(url=delete_url)
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

    # Launch label detection job
    def video__label_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)
        operation = self.clients["video"].annotate_video(
            request={
                "features": [videointelligence.Feature.LABEL_DETECTION],
                "input_uri": gcs_uri,
            }
        )
        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    # Launch text detection job
    def video__text_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)
        operation = self.clients["video"].annotate_video(
            request={
                "features": [videointelligence.Feature.TEXT_DETECTION],
                "input_uri": gcs_uri,
            }
        )

        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    # Launch face detection job
    def video__face_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)

        # Configure the request for each feature
        operation = self.clients["video"].annotate_video(
            request={
                "features": [videointelligence.Feature.FACE_DETECTION],
                "input_uri": gcs_uri,
                "video_context": videointelligence.VideoContext(
                    face_detection_config=videointelligence.FaceDetectionConfig(
                        include_bounding_boxes=True, include_attributes=True
                    )
                ),
            }
        )
        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    # Launch person tracking job
    def video__person_tracking_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)
        # Configure the request for each feature
        operation = self.clients["video"].annotate_video(
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
        )

        # Return job id (operation name)
        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    # Launch logo detection job
    def video__logo_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)
        # Configure the request for each feature
        operation = self.clients["video"].annotate_video(
            request={
                "features": [videointelligence.Feature.LOGO_RECOGNITION],
                "input_uri": gcs_uri,
            }
        )

        # Return job id (operation name)
        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    # Launch object tracking job
    def video__object_tracking_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)
        # Configure the request for each feature
        operation = self.clients["video"].annotate_video(
            request={
                "features": [videointelligence.Feature.OBJECT_TRACKING],
                "input_uri": gcs_uri,
            }
        )

        # Return job id (operation name)
        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    # Launch explicit content detection job
    def video__explicit_content_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)
        # Configure the request for each feature
        operation = self.clients["video"].annotate_video(
            request={
                "features": [videointelligence.Feature.EXPLICIT_CONTENT_DETECTION],
                "input_uri": gcs_uri,
            }
        )

        # Return job id (operation name)
        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    def video__shot_change_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        gcs_uri = self.google_upload_video(file=file)
        operation = self.clients["video"].annotate_video(
            request={
                "features": [videointelligence.Feature.SHOT_CHANGE_DETECTION],
                "input_uri": gcs_uri,
            }
        )

        return AsyncLaunchJobResponseType(provider_job_id=operation.operation.name)

    def video__label_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LabelDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)

        if result.get("done"):
            annotations = result["response"]["annotationResults"][0]
            label: List[dict] = annotations.get(
                "segmentLabelAnnotations", []
            ) + annotations.get("shotLabelAnnotations", [])
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
                        confidence=(sum(confidences) / len(confidences)),
                        timestamp=timestamps,
                    )
                )
            standardized_response = LabelDetectionAsyncDataClass(labels=label_list)

            return AsyncResponseType[LabelDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
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
            for annotation in annotations.get("textAnnotations", []):
                frames = []
                description = annotation["text"]
                for segment in annotation["segments"]:
                    confidence = round(segment["confidence"], 2)
                    for frame in segment["frames"]:
                        offset = frame["timeOffset"]
                        timestamp = float(offset[:-1])
                        xleft = frame["rotatedBoundingBox"]["vertices"][0].get("x", 0)
                        xright = frame["rotatedBoundingBox"]["vertices"][1].get("x", 0)
                        ytop = frame["rotatedBoundingBox"]["vertices"][0].get("y", 0)
                        ybottom = frame["rotatedBoundingBox"]["vertices"][2].get("y", 0)
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
            standardized_response = TextDetectionAsyncDataClass(texts=texts)
            return AsyncResponseType[TextDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
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
                for annotation in response["faceDetectionAnnotations"]:
                    for track in annotation["tracks"]:
                        timestamp = float(
                            track["timestampedObjects"][0]["timeOffset"][:-1]
                        )

                        top = float(
                            track["timestampedObjects"][0]["normalizedBoundingBox"].get(
                                "top", 0
                            )
                        )
                        left = float(
                            track["timestampedObjects"][0]["normalizedBoundingBox"].get(
                                "left", 0
                            )
                        )
                        right = float(
                            track["timestampedObjects"][0]["normalizedBoundingBox"].get(
                                "right", 0
                            )
                        )
                        bottom = float(
                            track["timestampedObjects"][0]["normalizedBoundingBox"].get(
                                "bottom", 0
                            )
                        )
                        # Bounding box
                        bounding_box = VideoBoundingBox(
                            top=top,
                            left=left,
                            height=1 - (top + (1 - bottom)),
                            width=1 - (left + (1 - right)),
                        )
                        attribute_dict = {}
                        for attr in track["timestampedObjects"][0].get(
                            "attributes", []
                        ):
                            attribute_dict[attr["name"]] = attr.get("confidence")
                        attributs = FaceAttributes(
                            headwear=attribute_dict.get("headwear"),
                            frontal_gaze=attribute_dict.get("looking_at_camera"),
                            eyes_visible=attribute_dict.get("eyes_visible"),
                            glasses=attribute_dict.get("glasses"),
                            mouth_open=attribute_dict.get("mouth_open"),
                            smiling=attribute_dict.get("smiling"),
                            brightness=None,
                            sharpness=None,
                            pose=VideoFacePoses(pitch=None, roll=None, yawn=None),
                        )
                        face = VideoFace(
                            offset=timestamp,
                            bounding_box=bounding_box,
                            attributes=attributs,
                            landmarks=LandmarksVideo(),
                        )
                        faces.append(face)
            standardized_response = FaceDetectionAsyncDataClass(faces=faces)
            return AsyncResponseType[FaceDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
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
            persons = response.get("personDetectionAnnotations")
            tracked_persons = []
            if persons:
                for person in persons:
                    tracked_person = []
                    for track in person["tracks"]:
                        for time_stamped_object in track["timestampedObjects"]:
                            top = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "top", 0
                                )
                            )
                            left = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "left", 0
                                )
                            )
                            right = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "right", 0
                                )
                            )
                            bottom = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "bottom", 0
                                )
                            )
                            # Bounding box
                            bounding_box = VideoTrackingBoundingBox(
                                top=top,
                                left=left,
                                height=1 - (top + (1 - bottom)),
                                width=1 - (left + (1 - right)),
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
                                shoulder_right=landmark_output.get(
                                    "right_shoulder", []
                                ),
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
                                    poses=VideoPersonPoses(
                                        pitch=None, roll=None, yaw=None
                                    ),
                                    quality=VideoPersonQuality(
                                        brightness=None, sharpness=None
                                    ),
                                )
                            )
                    tracked_persons.append(VideoTrackingPerson(tracked=tracked_person))
            standardized_response = PersonTrackingAsyncDataClass(
                persons=tracked_persons
            )

            return AsyncResponseType[PersonTrackingAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[PersonTrackingAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def video__logo_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LogoDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)

        if result.get("done"):
            response = result["response"]["annotationResults"][0]
            tracks = []
            if "logoRecognitionAnnotations" in response:
                for logo in response["logoRecognitionAnnotations"]:
                    objects = []
                    description = logo["entity"]["description"]
                    for track in logo["tracks"]:
                        for time_stamped_object in track["timestampedObjects"]:
                            timestamp = float(time_stamped_object["timeOffset"][:-1])
                            top = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "top", 0
                                )
                            )
                            left = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "left", 0
                                )
                            )
                            right = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "right", 0
                                )
                            )
                            bottom = float(
                                time_stamped_object["normalizedBoundingBox"].get(
                                    "bottom", 0
                                )
                            )
                            # Bounding box
                            bounding_box = VideoLogoBoundingBox(
                                top=top,
                                left=left,
                                height=1 - (top + (1 - bottom)),
                                width=1 - (left + (1 - right)),
                            )

                            objects.append(
                                VideoLogo(
                                    timestamp=timestamp,
                                    bounding_box=bounding_box,
                                    confidence=track.get("confidence", 0),
                                )
                            )
                    tracks.append(LogoTrack(description=description, tracking=objects))
            standardized_response = LogoDetectionAsyncDataClass(logos=tracks)

            return AsyncResponseType[LogoDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
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
                confidence = detected_object.get("confidence", 0) / 100
                description = detected_object["entity"]["description"]
                for frame in detected_object["frames"]:
                    timestamp = float(frame["timeOffset"][:-1])
                    top = float(frame["normalizedBoundingBox"].get("top", 0))
                    left = float(frame["normalizedBoundingBox"].get("left", 0))
                    right = float(frame["normalizedBoundingBox"].get("right", 0))
                    bottom = float(frame["normalizedBoundingBox"].get("bottom", 0))
                    # Bounding box
                    bounding_box = VideoObjectBoundingBox(
                        top=top,
                        left=left,
                        height=1 - (top + (1 - bottom)),
                        width=1 - (left + (1 - right)),
                    )
                    frames.append(
                        ObjectFrame(timestamp=timestamp, bounding_box=bounding_box)
                    )
                object_tracking.append(
                    ObjectTrack(
                        description=description, confidence=confidence, frames=frames
                    )
                )
            standardized_response = ObjectTrackingAsyncDataClass(
                objects=object_tracking
            )
            return AsyncResponseType[ObjectTrackingAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[ObjectTrackingAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def video__explicit_content_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[ExplicitContentDetectionAsyncDataClass]:
        result = google_video_get_job(provider_job_id)

        if result.get("error"):
            raise ProviderException(result["error"].get("message"))

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
            standardized_response = ExplicitContentDetectionAsyncDataClass(
                moderation=label_list
            )
            return AsyncResponseType[ExplicitContentDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[ExplicitContentDetectionAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def video__shot_change_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[ShotChangeDetectionAsyncDataClass]:

        result = google_video_get_job(provider_job_id)
        print(result)
        if result.get("done"):
            response = result["response"]["annotationResults"][0]
            shot_annotations = response.get("shotAnnotations", [])
            shots = []
            for shot in shot_annotations:
                start = float(shot["startTimeOffset"][:-1])
                end = float(shot["endTimeOffset"][:-1])
                shots.append(ShotFrame(startTimeOffset=start, endTimeOffset=end))
            standardized_response = ShotChangeDetectionAsyncDataClass(
                shotAnnotations=shots
            )

            return AsyncResponseType[ShotChangeDetectionAsyncDataClass](
                status="succeeded",
                original_response=result["response"],
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[ShotChangeDetectionAsyncDataClass](
            status="pending", provider_job_id=provider_job_id
        )

    def _bytes_to_mega(self, bytes_value):
        return bytes_value / (1024 * 1024)

    def _request_question_answer(self, model, api_key, text, temperature, file_data):
        base_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        url = base_url.format(model=model, api_key=api_key)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": text},
                        {
                            "file_data": {
                                "mime_type": file_data["mimeType"],
                                "file_uri": file_data["uri"],
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"candidateCount": 1, "temperature": temperature},
        }
        response = requests.post(url, json=payload)
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            self._delete_file(file=file_data["name"], api_key=api_key)
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        if response.status_code != 200:
            self._delete_file(file=file_data["name"], api_key=api_key)
            raise ProviderException(
                message=original_response["error"]["message"],
                code=response.status_code,
            )
        generated_text = original_response["candidates"][0]["content"]["parts"][0][
            "text"
        ]
        calculate_usage_tokens(original_response=original_response)
        return original_response, generated_text

    def video__question_answer(
        self,
        text: str,
        file: str,
        file_url: str = "",
        temperature: float = 0,
        model: str = None,
    ) -> QuestionAnswerDataClass:
        api_key = self.api_settings.get("genai_api_key")
        file_data = self._upload_and_process_file(file, api_key)
        file_size_mb = self._bytes_to_mega(int(file_data.get("sizeBytes", 0)))
        if file_size_mb >= 10:
            self._delete_file(file=file_data["name"], api_key=api_key)
            raise ProviderException(
                message="The video file is too large (over 10 MB). Please use the asynchronous video question answering api instead.",
            )
        if file_data["state"] == "PROCESSING":
            sleep(5)
            file_data = self._check_file_status(file_data["uri"], api_key)

        original_response, generated_text = self._request_question_answer(
            model=model,
            api_key=api_key,
            text=text,
            temperature=temperature,
            file_data=file_data,
        )

        self._delete_file(file=file_data["name"], api_key=api_key)
        return ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=QuestionAnswerDataClass(answer=generated_text),
        )

    def video__question_answer_async__launch_job(
        self,
        text: str,
        file: str,
        file_url: str = "",
        temperature: float = 0,
        model: str = None,
    ) -> AsyncLaunchJobResponseType:
        data_job_id = {}
        api_key = self.api_settings.get("genai_api_key")
        file_data = self._upload_and_process_file(file, api_key)
        job_id = file_data["name"].split("/")[1]
        inputs = {
            "text": text,
            "temperature": temperature,
            "model": model,
        }
        data_job_id[job_id] = inputs
        requests.post(
            url=f"https://webhook.site/{self.webhook_token}",
            data=json.dumps(data_job_id),
            headers={"content-type": "application/json"},
        )

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def video__question_answer_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType:
        api_key = self.api_settings.get("genai_api_key")
        url = f"https://generativelanguage.googleapis.com/v1beta/files/{provider_job_id}?key={api_key}"
        response = requests.get(url=url)
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        try:
            file_data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        if file_data["state"] == "PROCESSING":
            return AsyncPendingResponseType[QuestionAnswerAsyncDataClass](
                status="pending", provider_job_id=provider_job_id
            )
        else:
            wehbook_result, _ = check_webhook_result(
                provider_job_id, self.webhook_settings
            )
            result_object = (
                next(
                    filter(
                        lambda response: provider_job_id in response["content"],
                        wehbook_result,
                    ),
                    None,
                )
                if wehbook_result
                else None
            )
            try:
                content = json.loads(result_object["content"]).get(
                    provider_job_id, None
                )
            except json.JSONDecodeError:
                raise ProviderException("An error occurred while parsing the response.")

            original_response, generated_text = self._request_question_answer(
                model=content["model"],
                api_key=api_key,
                text=content["text"],
                temperature=content["temperature"],
                file_data=file_data,
            )
            self._delete_file(file=file_data["name"], api_key=api_key)
            return AsyncResponseType[QuestionAnswerAsyncDataClass](
                original_response=original_response,
                standardized_response=QuestionAnswerAsyncDataClass(
                    answer=generated_text
                ),
                provider_job_id=provider_job_id,
            )
