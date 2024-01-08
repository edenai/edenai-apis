from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import StreamChat, ChatStreamResponse
from typing import Dict, List, Optional, Union, Generator
import requests
import json



class PerplexityApi() :  

    def __init__(self, api_keys: Dict = {}) :
        self.provider_name = "perplexityai"
        self.url = "https://api.perplexity.ai"

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization" : f"Bearer {self.api_settings['api_key']}"
        }       
    
    def __text_to_json (self, lst_data : List[str]) -> Generator[ChatStreamResponse, None, None]:
        lenght = len(lst_data)
        i = 0
        while i < lenght :
            if (lst_data[i].startswith('data:')) :
                lst_data[i] = lst_data[i].replace('data: ', '')
                lst_data[i] = lst_data[i].replace('\r', '')
                i += 1  
            else :
                lst_data.pop(i)
            lenght = len(lst_data)
        for i in range (len(lst_data)) :
            jsonres = json.loads(lst_data[i])
            yield ChatStreamResponse(
                text = jsonres['choices'][0]['delta']['content'],
                blocked=False,
                provider='perplexityai'
            )
    
    def text__chat(
             self,
                text: str,
                chatbot_global_action: Optional[str] = None,
                previous_history: Optional[List[Dict[str, str]]]= None,
                temperature: float = 0.0,
                max_tokens: int = 64,
                model: Optional[str] = "pplx-7b-online",
                stream: bool = False,
                ) ->ResponseType[Union[ChatDataClass, StreamChat]]:

        messages = [{"role": "user", "content": text}]  
        
        if previous_history:
            for message in previous_history:
                messages.append(
                    {"role": message.get("role"), "content": message.get("message")},
                )

        if chatbot_global_action:
            messages.append({"role": "system", "content": chatbot_global_action})


        url = f"{self.url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        if (response.status_code != 200) :
                raise ProviderException(response.text, response.status_code)

        elif (response.status_code == 200) :
            if stream == False :
                try:
                    original_response = response.json()
                except requests.JSONDecodeError:
                    raise ProviderException(response.text, code = response.status_code)
        
        if stream == False :
            generated_text=original_response['choices'][0]['message']['content']
            message = [
                ChatMessageDataClass(role="user", message=text),
                ChatMessageDataClass(role="system", message=generated_text),
            ]
            standardized_response = ChatDataClass(generated_text=generated_text, message=message)

            
            return ResponseType[ChatDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
            )

        elif stream == True :
            data = response.text
            lst_data = []
            lst_data = data.split("\n")
            return ResponseType[StreamChat] (
                original_response=None, 
                standardized_response=StreamChat(stream=self.__text_to_json(lst_data))
            )
        
            
