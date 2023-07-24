from collections import defaultdict
from io import BufferedReader

from edenai_apis.features.video.explicit_content_detection_async.explicit_content_detection_async_dataclass import (
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
    VideoLabelBoundingBox,
    VideoLabelTimeStamp,
)
from edenai_apis.features.video.person_tracking_async.person_tracking_async_dataclass import (
    PersonLandmarks,
    PersonTracking,
    PersonTrackingAsyncDataClass,
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
from edenai_apis.features.video.video_interface import VideoInterface
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType

from .helpers import (
    amazon_launch_video_job,
    amazon_video_original_response,
    amazon_video_response_formatter,
)


class AmazonVideoApi(VideoInterface):
    # Launch job label detection
    def video__label_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(
            provider_job_id=amazon_launch_video_job(file, "LABEL")
        )

    # Launch job text detection
    def video__text_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(
            provider_job_id=amazon_launch_video_job(file, "TEXT")
        )

    # Launch job face detection
    def video__face_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(
            provider_job_id=amazon_launch_video_job(file, "FACE")
        )

    # Launch job person tracking
    def video__person_tracking_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(
            provider_job_id=amazon_launch_video_job(file, "PERSON")
        )

    # Launch job explicit content detection
    def video__explicit_content_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(
            provider_job_id=amazon_launch_video_job(file, "EXPLICIT")
        )

    # Get job result for label detection
    def video__label_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LabelDetectionAsyncDataClass]:
        pagination_token = ""
        max_result = 20
        finished = False
        while not finished:
            response = amazon_video_original_response(
                provider_job_id,
                max_result,
                pagination_token,
                self.clients["video"].get_label_detection,
                "TIMESTAMP",
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
                        top=instance["BoundingBox"].get("Top", 0),
                        left=instance["BoundingBox"].get("Left", 0),
                        width=instance["BoundingBox"].get("Width", 0),
                        height=instance["BoundingBox"].get("Height", 0),
                    )
                    boxes.append(video_box)

                videolabel = VideoLabel(
                    timestamp=[
                        VideoLabelTimeStamp(
                            start=float(label["Timestamp"]) / 1000.0, end=None
                        )
                    ],
                    confidence=label["Label"]["Confidence"],
                    name=label["Label"]["Name"],
                    category=parents,
                    bounding_box=boxes,
                )
                labels.append(videolabel)

            standardized_response = LabelDetectionAsyncDataClass(labels=labels)
            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standardized_response, provider_job_id
        )

    # Get job result for text detection
    def video__text_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> TextDetectionAsyncDataClass:
        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = amazon_video_original_response(
                provider_job_id,
                max_results,
                pagination_token,
                self.clients["video"].get_text_detection,
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
                    confidence = round(
                        annotation["TextDetection"]["Confidence"] / 100, 2
                    )
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

            standardized_response = TextDetectionAsyncDataClass(texts=text_video)

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standardized_response, provider_job_id
        )

    # Get job result for face detection
    def video__face_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> FaceDetectionAsyncDataClass:
        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = amazon_video_original_response(
                provider_job_id,
                max_results,
                pagination_token,
                self.clients["video"].get_face_detection,
            )

            faces = []
            for face in response["Faces"]:
                # Time stamp
                offset = float(face["Timestamp"]) / 1000.0  # convert to seconds

                # Bounding box
                bounding_box = VideoBoundingBox(
                    top=face["Face"]["BoundingBox"].get("Top", 0),
                    left=face["Face"]["BoundingBox"].get("Left", 0),
                    height=face["Face"]["BoundingBox"].get("Height", 0),
                    width=face["Face"]["BoundingBox"].get("Width", 0),
                )

                # Attributes
                poses = VideoFacePoses(
                    pitch=face["Face"]["Pose"].get("Pitch", 0) / 100,
                    yawn=face["Face"]["Pose"].get("Yaw", 0) / 100,
                    roll=face["Face"]["Pose"].get("Roll", 0) / 100,
                )
                attributes_video = FaceAttributes(
                    pose=poses,
                    brightness=face["Face"]["Quality"].get("Brightness", 0) / 100,
                    sharpness=face["Face"]["Quality"].get("Sharpness", 0) / 100,
                    headwear=None,
                    frontal_gaze=None,
                    eyes_visible=None,
                    glasses=None,
                    mouth_open=None,
                    smiling=None,
                )

                # Landmarks
                landmarks_output = {}
                for land in face["Face"]["Landmarks"]:
                    if land.get("Type") and land.get("X") and land.get("Y"):
                        landmarks_output[land["Type"]] = [land["X"], land["Y"]]

                landmarks_video = LandmarksVideo(
                    eye_left=landmarks_output.get("eyeLeft", []),
                    eye_right=landmarks_output.get("eyeRight", []),
                    mouth_left=landmarks_output.get("mouthLeft", []),
                    mouth_right=landmarks_output.get("mouthRight", []),
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
            standardized_response = FaceDetectionAsyncDataClass(faces=faces)

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standardized_response, provider_job_id
        )

    # Get job result for person tracking
    def video__person_tracking_async__get_job_result(
        self, provider_job_id: str
    ) -> PersonTrackingAsyncDataClass:
        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = amazon_video_original_response(
                provider_job_id,
                max_results,
                pagination_token,
                self.clients["video"].get_person_tracking,
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
                            top=bounding_box.get("Top", 0),
                            left=bounding_box.get("Left", 0),
                            height=bounding_box.get("Height", 0) ,
                            width=bounding_box.get("Width", 0),
                        )
                        face = detected_person["Person"].get("Face")
                        # Get landmarks
                        poses = VideoPersonPoses.default()
                        landmarks = PersonLandmarks()
                        quality = VideoPersonQuality.default()
                        if face:
                            landmarks_dict = {}
                            for land in face.get("Landmarks", []):
                                landmarks_dict[land["Type"]] = [land["X"], land["Y"]]
                            landmarks = PersonLandmarks(
                                eye_left=landmarks_dict.get("eyeLeft", []),
                                eye_right=landmarks_dict.get("eyeRight", []),
                                nose=landmarks_dict.get("nose", []),
                                mouth_left=landmarks_dict.get("mouthLeft", []),
                                mouth_right=landmarks_dict.get("mouthRight", []),
                            )
                            poses = VideoPersonPoses(
                                roll=face.get("Pose").get("Roll"),
                                yaw=face.get("Pose").get("Yaw"),
                                pitch=face.get("Pose").get("Pitch"),
                            )
                            quality = VideoPersonQuality(
                                brightness=face.get("Quality").get("Brightness"),
                                sharpness=face.get("Quality").get("Sharpness"),
                            )

                        tracked_person.append(
                            PersonTracking(
                                offset=offset,
                                bounding_box=bounding_box,
                                landmarks=landmarks,
                                poses=poses,
                                quality=quality,
                            )
                        )
                tracked_persons.append(VideoTrackingPerson(tracked=tracked_person))
            standardized_response = PersonTrackingAsyncDataClass(
                persons=tracked_persons
            )

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standardized_response, provider_job_id
        )

    # Get job result for explicit content detection
    def video__explicit_content_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> ExplicitContentDetectionAsyncDataClass:
        max_results = 10
        pagination_token = ""
        finished = False

        while not finished:
            response = amazon_video_original_response(
                provider_job_id,
                max_results,
                pagination_token,
                self.clients["video"].get_content_moderation,
            )

            moderated_content = []
            for label in response.get("ModerationLabels", []):
                confidence = label.get("ModerationLabel", defaultdict).get("Confidence")
                timestamp = float(label.get("Timestamp")) / 1000.0  # convert to seconds
                if label.get("ParentName"):
                    category = label.get("ParentName", label.get("Name"))
                    moderated_content.append(
                        ContentNSFW(
                            timestamp=timestamp,
                            confidence=confidence,
                            category=category,
                        )
                    )
            standardized_response = ExplicitContentDetectionAsyncDataClass(
                moderation=moderated_content
            )
            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return amazon_video_response_formatter(
            response, standardized_response, provider_job_id
        )
