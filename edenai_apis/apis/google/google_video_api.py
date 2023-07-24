from pathlib import Path
from time import time
from io import BufferedReader
from edenai_apis.apis.google.google_helpers import GoogleVideoFeatures
from edenai_apis.features.video.video_interface import VideoInterface
from edenai_apis.utils.types import AsyncLaunchJobResponseType

from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.apis.google.google_helpers import (
    GoogleVideoFeatures,
    google_video_get_job,
    score_to_content,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.features.video import (
    ContentNSFW,
    ExplicitContentDetectionAsyncDataClass,
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

from google.cloud import videointelligence


class GoogleVideoApi(VideoInterface):
    def google_video_launch_job(
        self,
        file: str,
        feature: GoogleVideoFeatures,
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
        return AsyncLaunchJobResponseType(
            provider_job_id=features[feature].operation.name
        )

    # Launch label detection job
    def video__label_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.LABEL)

    # Launch text detection job
    def video__text_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.TEXT)

    # Launch face detection job
    def video__face_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.FACE)

    # Launch person tracking job
    def video__person_tracking_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.PERSON)

    # Launch logo detection job
    def video__logo_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.LOGO)

    # Launch object tracking job
    def video__object_tracking_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.OBJECT)

    # Launch explicit content detection job
    def video__explicit_content_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return self.google_video_launch_job(file, GoogleVideoFeatures.EXPLICIT)

    def video__label_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LabelDetectionAsyncDataClass]:
        try:
            result = google_video_get_job(provider_job_id)
        except ProviderException as provider_excp:
            raise provider_excp
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        if result.get("done"):
            annotations = result["response"]["annotationResults"][0]
            label = annotations.get("segmentLabelAnnotations", "") + annotations.get(
                "shotLabelAnnotations", ""
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
        try:
            result = google_video_get_job(provider_job_id)
        except ProviderException as provider_excp:
            raise provider_excp
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        if result.get("done"):
            annotations = result["response"]["annotationResults"][0]
            texts = []
            for annotation in annotations["textAnnotations"]:
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
        try:
            result = google_video_get_job(provider_job_id)
        except ProviderException as provider_excp:
            raise provider_excp
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        if result.get("done"):
            faces = []
            response = result["response"]["annotationResults"][0]
            if response.get("faceDetectionAnnotations") is not None:
                for annotation in response["faceDetectionAnnotations"]:
                    for track in annotation["tracks"]:
                        timestamp = float(
                            track["timestampedObjects"][0]["timeOffset"][:-1]
                        )
                        bounding_box = VideoBoundingBox(
                            top=track["timestampedObjects"][0][
                                "normalizedBoundingBox"
                            ].get("top", 0),
                            left=track["timestampedObjects"][0][
                                "normalizedBoundingBox"
                            ].get("left", 0),
                            height=track["timestampedObjects"][0][
                                "normalizedBoundingBox"
                            ].get("bottom", 0),
                            width=track["timestampedObjects"][0][
                                "normalizedBoundingBox"
                            ].get("right", 0),
                        )
                        attribute_dict = {}
                        for attr in track["timestampedObjects"][0].get(
                            "attributes", []
                        ):
                            attribute_dict[attr["name"]] = attr["confidence"]
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
        try:
            result = google_video_get_job(provider_job_id)
        except ProviderException as provider_excp:
            raise provider_excp
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

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
                                time_stamped_object["normalizedBoundingBox"].get("top",0)
                            ),
                            left=float(
                                time_stamped_object["normalizedBoundingBox"].get("left",0)
                            ),
                            height=float(
                                time_stamped_object["normalizedBoundingBox"].get("bottom",0)
                            ),
                            width=float(
                                time_stamped_object["normalizedBoundingBox"].get("right",0)
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
                                poses=VideoPersonPoses(pitch=None, roll=None, yaw=None),
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
        try:
            result = google_video_get_job(provider_job_id)
        except ProviderException as provider_excp:
            raise provider_excp
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

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
                            bounding_box = VideoLogoBoundingBox(
                                top=time_stamped_object["normalizedBoundingBox"].get("top", 0),
                                left=time_stamped_object["normalizedBoundingBox"].get("left", 0),
                                height=time_stamped_object["normalizedBoundingBox"].get("bottom", 0),
                                width=time_stamped_object["normalizedBoundingBox"].get("right", 0),
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
        try:
            result = google_video_get_job(provider_job_id)
        except ProviderException as provider_excp:
            raise provider_excp
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

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
                    bounding_box = VideoObjectBoundingBox(
                        top=float(frame["normalizedBoundingBox"].get("top", 0)),
                        left=float(frame["normalizedBoundingBox"].get("left", 0)),
                        width=float(frame["normalizedBoundingBox"].get("right", 0)),
                        height=float(frame["normalizedBoundingBox"].get("bottom", 0)),
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
        try:
            result = google_video_get_job(provider_job_id)
        except ProviderException as provider_excp:
            raise provider_excp
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

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
