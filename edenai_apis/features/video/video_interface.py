from abc import abstractmethod
from io import BufferedReader

from edenai_apis.features.video import (
    ExplicitContentDetectionAsyncDataClass,
    FaceDetectionAsyncDataClass,
    LabelDetectionAsyncDataClass,
    LogoDetectionAsyncDataClass,
    ObjectTrackingAsyncDataClass,
    PersonTrackingAsyncDataClass,
    TextDetectionAsyncDataClass,
)
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType

class VideoInterface:
    ### Explicit content detection methods
    @abstractmethod
    def video__explicit_content_detection_async__launch_job(
        self, file: BufferedReader
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

    ### Face detection methods
    @abstractmethod
    def video__face_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
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
    def video__label_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
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
    def video__logo_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
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
    def video__object_tracking_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
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
    def video__person_tracking_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
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
    def video__text_detection_async__launch_job(self, file: BufferedReader) -> AsyncLaunchJobResponseType:
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
