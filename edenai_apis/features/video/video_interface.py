from abc import ABC, abstractmethod
from io import BufferedReader
from typing import Optional

from edenai_apis.features.video import (
    ExplicitContentDetectionAsyncDataClass,
    FaceDetectionAsyncDataClass,
    LabelDetectionAsyncDataClass,
    LogoDetectionAsyncDataClass,
    ObjectTrackingAsyncDataClass,
    PersonTrackingAsyncDataClass,
    TextDetectionAsyncDataClass,
    QuestionAnswerDataClass,
    GenerationAsyncDataClass,
)
from edenai_apis.features.video.deepfake_detection_async.deepfake_detection_async_dataclass import (
    DeepfakeDetectionAsyncDataClass,
)
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType


class VideoInterface:
    ### Explicit content detection methods
    @abstractmethod
    def video__explicit_content_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to detect explicit content in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__explicit_content_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[ExplicitContentDetectionAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    ### Deepfake detection methods
    @abstractmethod
    def video__deepfake_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to detect altered videos via inconsistencies

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__deepfake_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[DeepfakeDetectionAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    ### Face detection methods
    @abstractmethod
    def video__face_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to detect faces in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__face_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[FaceDetectionAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    ### Label detection methods
    @abstractmethod
    def video__label_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to detect objects in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__label_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LabelDetectionAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    ### Logo detection methods
    @abstractmethod
    def video__logo_detection_async__launch_job(
        self, file: str, file_url: str = "", language: str = "en", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to detect logos in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__logo_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LogoDetectionAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    ### Object tracking methods
    @abstractmethod
    def video__object_tracking_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to track objects in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__object_tracking_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[ObjectTrackingAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    ### Person tracking methods
    @abstractmethod
    def video__person_tracking_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to track persons in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__person_tracking_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[PersonTrackingAsyncDataClass]:
        raise NotImplementedError

    ### Text detection methods
    @abstractmethod
    def video__text_detection_async__launch_job(
        self, file: str, file_url: str = "", language: str = "en", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to detect text in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__text_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[TextDetectionAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    @abstractmethod
    def video__shot_change_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        raise NotImplementedError

    @abstractmethod
    def video__shot_change_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType:
        raise NotImplementedError

    @abstractmethod
    def video__question_answer(
        self,
        text: str,
        file: str,
        file_url: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> QuestionAnswerDataClass:
        raise NotImplementedError

    @abstractmethod
    def video__question_answer_async__launch_job(
        self,
        text: str,
        file: str,
        file_url: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        raise NotImplementedError

    @abstractmethod
    def video__question_answer_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType:
        raise NotImplementedError

    ### Video generation methods
    @abstractmethod
    def video__generation_async__launch_job(
        self,
        text: str,
        duration: Optional[int] = 6,
        fps: Optional[int] = 24,
        dimension: Optional[str] = "1280x720",
        seed: Optional[float] = 12,
        file: Optional[str] = None,
        file_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        """
        Launch an asynchronous job to detect text in a video

        Args:
            file (BufferedReader): video to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def video__generation_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[GenerationAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError
