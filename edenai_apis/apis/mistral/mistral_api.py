from typing import Dict
from google.protobuf.json_format import MessageToDict
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api.status import status_code_pb2

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text.generation.generation_dataclass import (
    GenerationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType



class MistralApi(ProviderInterface, TextInterface):
    provider_name = "mistral"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.user_id = self.api_settings["user_id"]
        self.app_id = self.api_settings["app_id"]
        self.key = self.api_settings["key"]

    def __chat_markup_tokens(self, model):
        if model == "mistral-7B-Instruct":
            return "[INST]", "[/INST]"
        else:
            return "<|im_start|>", "<|im_end|>"
    
    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str
    ) -> ResponseType[GenerationDataClass]:
        start, end = self.__chat_markup_tokens(model)
        
        text = f"{start} {text} {end}"

        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id="mistralai", app_id="completion"
        )

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=user_data_object,
                model_id=model,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw=text))
                    )
                ],
            ),
            metadata=metadata,
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise ProviderException(
                post_model_outputs_response.status.description,
                code=post_model_outputs_response.status.code,
            )

        response = MessageToDict(
            post_model_outputs_response, preserving_proto_field_name=True
        )

        output = response.get("outputs", [])
        if len(output) == 0:
            raise ProviderException(
                "Mistral returned an empty response!",
                code=post_model_outputs_response.status.code,
            )

        original_response = output[0].get("data", {}) or {}

        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(
                generated_text=(original_response.get("text", {}) or {}).get("raw", "")
            ),
        )
