from typing import Dict
from edenai_apis.features.text import GenerationDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.utils.types import ResponseType
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from edenai_apis.utils.exception import ProviderException
from google.protobuf.json_format import MessageToDict

class AnthropicApi(
    ProviderInterface,
    TextInterface

):
    provider_name = "anthropic"
    
    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.user_id = "anthropic"
        self.app_id = "completion"
        self.key = self.api_settings["api_key"]
    
    def text__generation(
        self, 
        text: str,
        temperature: float, 
        max_tokens: int,
        model: str,) -> ResponseType[GenerationDataClass]:
        
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        metadata = (("authorization", self.key),)
        userDataObject = resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id)
        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=userDataObject, 
                model_id=model,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                             text=resources_pb2.Text(
                                raw=text
                            )
                        )
                    )
                ]
            ),
            metadata=metadata
        )
        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise ProviderException(
                post_model_outputs_response.status.description,
                code= post_model_outputs_response.status.code
                )
        response = MessageToDict(
            post_model_outputs_response, preserving_proto_field_name=True
        )
        output = response.get("outputs", [])
        if len(output) == 0:
            raise ProviderException(
                "Anthropic returned an empty response!",
                code= post_model_outputs_response.status.code
                )
        original_response = output[0].get("data", {}) or {}
        
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(generated_text=original_response.get('text', {}).get('raw', '')),
        )