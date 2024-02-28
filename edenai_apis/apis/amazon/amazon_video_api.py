from edenai_apis.features.video.explicit_content_detection_async.explicit_content_detection_async_dataclass import (
    ExplicitContentDetectionAsyncDataClass,
)
from edenai_apis.features.video.face_detection_async.face_detection_async_dataclass import (
    FaceDetectionAsyncDataClass,
)
from edenai_apis.features.video.label_detection_async.label_detection_async_dataclass import (
    LabelDetectionAsyncDataClass,
)
from edenai_apis.features.video.person_tracking_async.person_tracking_async_dataclass import (
    PersonTrackingAsyncDataClass,
)
from edenai_apis.features.video.text_detection_async.text_detection_async_dataclass import (
    TextDetectionAsyncDataClass,
)
from edenai_apis.features.video.video_interface import VideoInterface
from edenai_apis.utils.exception import (
    ProviderException,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from .helpers import (
    amazon_get_video_data,
    handle_amazon_call,
    amazon_video_person_tracking_parser,
    amazon_video_labels_parser,
    amazon_video_text_parser,
    amazon_video_face_parser,
    amazon_video_explicit_parser,
)
from .config import clients


class AmazonVideoApi(VideoInterface):
    # Launch job label detection
    def video__label_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        video, notification_channel = amazon_get_video_data(file=file)
        response = clients(self.api_settings)["video"].start_label_detection(
            Video=video, NotificationChannel=notification_channel
        )
        # return job id
        job_id = response["JobId"]

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def video__text_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        video, notification_channel = amazon_get_video_data(file=file)
        response = clients(self.api_settings)["video"].start_text_detection(
            Video=video, NotificationChannel=notification_channel
        )
        # return job id
        job_id = response["JobId"]

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    # Launch job face detection
    def video__face_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        video, notification_channel = amazon_get_video_data(file=file)
        response = clients(self.api_settings)["video"].start_face_detection(
            Video=video, NotificationChannel=notification_channel
        )
        # return job id
        job_id = response["JobId"]

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    # Launch job person tracking
    def video__person_tracking_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        video, notification_channel = amazon_get_video_data(file=file)
        response = clients(self.api_settings)["video"].start_person_tracking(
            Video=video, NotificationChannel=notification_channel
        )
        # return job id
        job_id = response["JobId"]

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    # Launch job explicit content detection
    def video__explicit_content_detection_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        video, notification_channel = amazon_get_video_data(file=file)
        response = clients(self.api_settings)["video"].start_content_moderation(
            Video=video, NotificationChannel=notification_channel
        )
        # return job id
        job_id = response["JobId"]

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    # Get job result for label detection
    def video__label_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LabelDetectionAsyncDataClass]:
        payload = {"JobId": provider_job_id}
        response = handle_amazon_call(
            self.clients["video"].get_label_detection, **payload
        )
        if response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        if response["JobStatus"] == "SUCCEEDED":
            pagination_token = response.get("NextToken")
            responses = [response]
            while pagination_token:
                payload = {
                    "JobId": provider_job_id,
                    "NextToken": pagination_token,
                }
                response = handle_amazon_call(
                    self.clients["video"].get_label_detection, **payload
                )

                if response["JobStatus"] == "FAILED":
                    error: str = response.get(
                        "StatusMessage", "Amazon returned a job status: FAILED"
                    )
                    raise ProviderException(error)

                responses.append(response)
                pagination_token = response.get("NextToken")

            labels = []
            for response in responses:
                labels.extend(amazon_video_labels_parser(response))

            return AsyncResponseType(
                original_response=responses,
                standardized_response=LabelDetectionAsyncDataClass(labels=labels),
                provider_job_id=provider_job_id,
            )
        return AsyncPendingResponseType(provider_job_id=response["JobStatus"])

    # Get job result for text detection
    def video__text_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> TextDetectionAsyncDataClass:
        payload = {"JobId": provider_job_id}
        response = handle_amazon_call(
            self.clients["video"].get_text_detection, **payload
        )
        if response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        if response["JobStatus"] == "SUCCEEDED":
            pagination_token = response.get("NextToken")
            responses = [response]
            while pagination_token:
                payload = {
                    "JobId": provider_job_id,
                    "NextToken": pagination_token,
                }
                response = handle_amazon_call(
                    self.clients["video"].get_text_detection, **payload
                )

                if response["JobStatus"] == "FAILED":
                    error: str = response.get(
                        "StatusMessage", "Amazon returned a job status: FAILED"
                    )
                    raise ProviderException(error)

                responses.append(response)
                pagination_token = response.get("NextToken")

            texts = []
            for response in responses:
                texts.extend(amazon_video_text_parser(response))

            return AsyncResponseType(
                original_response=responses,
                standardized_response=TextDetectionAsyncDataClass(texts=texts),
                provider_job_id=provider_job_id,
            )
        return AsyncPendingResponseType(provider_job_id=response["JobStatus"])

    # Get job result for face detection
    def video__face_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> FaceDetectionAsyncDataClass:
        payload = {"JobId": provider_job_id}
        response = handle_amazon_call(
            self.clients["video"].get_face_detection, **payload
        )
        if response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        if response["JobStatus"] == "SUCCEEDED":
            pagination_token = response.get("NextToken")
            responses = [response]
            while pagination_token:
                payload = {
                    "JobId": provider_job_id,
                    "NextToken": pagination_token,
                }
                response = handle_amazon_call(
                    self.clients["video"].get_face_detection, **payload
                )

                if response["JobStatus"] == "FAILED":
                    error: str = response.get(
                        "StatusMessage", "Amazon returned a job status: FAILED"
                    )
                    raise ProviderException(error)

                responses.append(response)
                pagination_token = response.get("NextToken")

            faces = []
            for response in responses:
                faces.extend(amazon_video_face_parser(response))

            return AsyncResponseType(
                original_response=responses,
                standardized_response=FaceDetectionAsyncDataClass(faces=faces),
                provider_job_id=provider_job_id,
            )
        return AsyncPendingResponseType(provider_job_id=response["JobStatus"])

    # Get job result for person tracking
    def video__person_tracking_async__get_job_result(
        self, provider_job_id: str
    ) -> PersonTrackingAsyncDataClass:
        payload = {"JobId": provider_job_id}
        response = handle_amazon_call(
            self.clients["video"].get_person_tracking, **payload
        )
        if response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        if response["JobStatus"] == "SUCCEEDED":
            pagination_token = response.get("NextToken")
            responses = [response]
            while pagination_token:
                payload = {
                    "JobId": provider_job_id,
                    "NextToken": pagination_token,
                }
                response = handle_amazon_call(
                    self.clients["video"].get_person_tracking, **payload
                )

                if response["JobStatus"] == "FAILED":
                    error: str = response.get(
                        "StatusMessage", "Amazon returned a job status: FAILED"
                    )
                    raise ProviderException(error)

                responses.append(response)
                pagination_token = response.get("NextToken")

            persons = []
            for response in responses:
                persons.extend(amazon_video_person_tracking_parser(response))

            return AsyncResponseType(
                original_response=responses,
                standardized_response=PersonTrackingAsyncDataClass(persons=persons),
                provider_job_id=provider_job_id,
            )
        return AsyncPendingResponseType(provider_job_id=response["JobStatus"])

    # Get job result for explicit content detection
    def video__explicit_content_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> ExplicitContentDetectionAsyncDataClass:
        payload = {"JobId": provider_job_id}
        response = handle_amazon_call(
            self.clients["video"].get_content_moderation, **payload
        )
        if response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        if response["JobStatus"] == "SUCCEEDED":
            pagination_token = response.get("NextToken")
            responses = [response]
            while pagination_token:
                payload = {
                    "JobId": provider_job_id,
                    "NextToken": pagination_token,
                }
                response = handle_amazon_call(
                    self.clients["video"].get_content_moderation, **payload
                )

                if response["JobStatus"] == "FAILED":
                    error: str = response.get(
                        "StatusMessage", "Amazon returned a job status: FAILED"
                    )
                    raise ProviderException(error)

                responses.append(response)
                pagination_token = response.get("NextToken")

            moderated_content = []
            for response in responses:
                moderated_content.extend(amazon_video_explicit_parser(response))

            return AsyncResponseType(
                original_response=responses,
                standardized_response=ExplicitContentDetectionAsyncDataClass(
                    moderation=moderated_content
                ),
                provider_job_id=provider_job_id,
            )
        return AsyncPendingResponseType(provider_job_id=response["JobStatus"])
