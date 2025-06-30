from typing import Optional
import base64
from io import BytesIO
import json

from edenai_apis.features.video import QuestionAnswerDataClass
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
from edenai_apis.features.video.generation_async.generation_async_dataclass import (
    GenerationAsyncDataClass,
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
    ResponseType,
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
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
)


class AmazonVideoApi(VideoInterface):
    # Launch job label detection
    def video__label_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        video, notification_channel = amazon_get_video_data(file=file)
        response = clients(self.api_settings)["video"].start_label_detection(
            Video=video, NotificationChannel=notification_channel
        )
        # return job id
        job_id = response["JobId"]

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def video__text_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
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
        self, file: str, file_url: str = "", **kwargs
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
        self, file: str, file_url: str = "", **kwargs
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
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        video, notification_channel = amazon_get_video_data(file=file)
        response = clients(self.api_settings)["video"].start_content_moderation(
            Video=video, NotificationChannel=notification_channel
        )
        # return job id
        job_id = response["JobId"]

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    # Launch job video generation
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
        text_input = {"text": text}
        if file:
            with open(file, "rb") as file_:
                file_content = file_.read()
                input_image_base64 = base64.b64encode(file_content).decode("utf-8")
                images = [{"format": "png", "source": {"bytes": input_image_base64}}]
                text_input["images"] = images
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": text_input,
            "videoGenerationConfig": {
                "durationSeconds": duration,
                "fps": fps,
                "dimension": dimension,
                "seed": seed,
            },
        }
        request_params = {
            "modelId": model,
            "modelInput": model_input,
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://us-storage"}},
        }
        response = handle_amazon_call(
            self.clients["bedrock"].start_async_invoke, **request_params
        )
        provider_job_id = response.get("invocationArn")
        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)

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

    # Get job result for generation
    def video__generation_async__get_job_result(
        self, provider_job_id: str
    ) -> GenerationAsyncDataClass:
        invocation = handle_amazon_call(
            self.clients["bedrock"].get_async_invoke,
            **{"invocationArn": provider_job_id},
        )
        if invocation["status"] == "Completed":
            file_name = invocation["outputDataConfig"]["s3OutputDataConfig"][
                "s3Uri"
            ].split("/")[-1]
            response = self.clients["s3"].get_object(
                Bucket="us-storage", Key=f"{file_name}/output.mp4"
            )
            data = response["Body"].read()
            video_content = base64.b64encode(data).decode("utf-8")
            resource_url = upload_file_bytes_to_s3(BytesIO(data), ".mp4", USER_PROCESS)
            return AsyncResponseType(
                original_response=invocation,
                standardized_response=GenerationAsyncDataClass(
                    video=video_content, video_resource_url=resource_url
                ),
                provider_job_id=provider_job_id,
            )
        if invocation["status"] == "InProgress":
            return AsyncPendingResponseType(provider_job_id=provider_job_id)

        if invocation["status"] == "Failed":
            failure_message = invocation["failureMessage"]
            raise ProviderException(failure_message)

    def video__question_answer(
        self,
        text: str,
        file: str,
        file_url: str = "",
        temperature: float = 0,
        max_tokens: int = None,
        model: str = None,
        **kwargs,
    ) -> QuestionAnswerDataClass:
        with open(file, "rb") as video_file:
            binary_data = video_file.read()
            base_64_encoded_data = base64.b64encode(binary_data)
            base64_string = base_64_encoded_data.decode("utf-8")
        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": "mp4",
                            "source": {"bytes": base64_string},
                        }
                    },
                    {"text": text},
                ],
            }
        ]
        request_params = {
            "body": json.dumps(
                {
                    "schemaVersion": "messages-v1",
                    "messages": message_list,
                    "inferenceConfig": {"temperature": temperature},
                }
            ),
            "modelId": model,
        }
        response = handle_amazon_call(
            self.clients["bedrock"].invoke_model, **request_params
        )
        model_response = json.loads(response["body"].read())
        usage = {
            "completion_tokens": model_response["usage"]["outputTokens"],
            "prompt_tokens": model_response["usage"]["inputTokens"],
            "total_tokens": model_response["usage"]["totalTokens"],
        }
        model_response["usage"] = usage
        content_text = model_response["output"]["message"]["content"][0]["text"]
        return ResponseType[QuestionAnswerDataClass](
            original_response=model_response,
            standardized_response=QuestionAnswerDataClass(answer=content_text),
            usage=usage,
        )
