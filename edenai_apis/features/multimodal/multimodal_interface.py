from abc import abstractmethod
from typing import Optional, List, Dict

from edenai_apis.features.multimodal.chat import ChatDataClass, ChatMessageDataClass
from edenai_apis.utils.types import ResponseType


class MultimodalInterface:

    @abstractmethod
    def multimodal__chat(
        self,
        messages: List[ChatMessageDataClass],
        chatbot_global_action: Optional[str],
        temperature: float = 0,
        max_tokens: int = 25,
        model: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        stream: bool = False,
        provider_params: Optional[dict] = None,
        response_format=None,
        **kwargs,
    ) -> ResponseType[ChatDataClass]:
        """
        Generate responses in a multimodal conversation using a chatbot.
        Args:
            messages (List[Dict[str, str]]): A list of messages exchanged in the conversation.
            chatbot_global_action (Optional[str]): The global action or context for the chatbot.
            temperature (float, optional): Controls the randomness of the response generation.
            max_tokens (int, optional): The maximum number of tokens to generate for each response.
            model (Optional[str], optional): The name or identifier of the model.
            stop_sequences (Optional[List[str]], optional): A list of strings that, if encountered
                in the generated response, will stop generation.
            top_k (Optional[int], optional): Controls the diversity of the generated responses
                by limiting the number of tokens considered at each step.
            top_p (Optional[int], optional): Controls the diversity of the generated responses
                by selecting from the most probable tokens whose cumulative probability exceeds
                the given value.
            stream (bool, optional): Whether to enable streaming for generating responses.
            provider_params (dict, optional): Additional parameters specific to the provider
        """
        raise NotImplementedError
