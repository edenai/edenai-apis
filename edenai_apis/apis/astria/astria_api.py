from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.image import GeneratedImageDataClass, GenerationDataClass


import requests
from typing import Dict, List, Optional, Literal
import uuid
import json
import time


class AstriaApi (ProviderInterface, ImageInterface) :
    provider_name = "astria"

    def __init__(self, api_keys : Dict = {})->None :
        
        self.url = "https://api.astria.ai/"

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        self.headers = {
            "authorization" : f"Bearer {self.api_settings['api_key']}"
        } 

    def wait_for_images(self, tuneid :int, promptid :int, num_images : int) ->List[str]:
        lst_url = self.image__get_image_prompt(tuneid, promptid)
        timewait = 0
        while (len(lst_url)!= num_images) :
            time.sleep(30)
            timewait += 30
            print(timewait)
            if timewait > 10000 :
                raise Exception("Timeout Error")
            else :
                lst_url = self.image__get_image_prompt(tuneid, promptid)
        return lst_url

    def image__promptinfo(self, tuneid : int, promptid : int)->json :
        response = requests.get(f"{self.url}tunes/{tuneid}/prompts/{promptid}", headers=self.headers)
        try :
            response = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)
        return response
    
    def image__get_image_prompt(self, tuneid : int, promptid : int) ->List[str]:
        promptinfo = self.image__promptinfo(tuneid, promptid)
        return promptinfo['images']

    def get__all_tunes(self) ->json:
        response = requests.get(f'{self.url}tunes', headers=self.headers)
        try : 
            response = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)
        return response
    
    def get_tune(self, tuneid : int = None)->json :
        if tuneid is not None :
            response = requests.get(f"{self.url}tunes/{tuneid}", headers=self.headers)
            try : 
                response = response.json()
            except json.JSONDecodeError :
                raise ProviderException(message=response.text, code=response.status_code)
            return response
        raise Exception("Missing id")


    def image__create_tune(self, name : str, description : str, images : List[str], token : Literal['owhx', 'sks'] = 'sks', base_tune_id : Optional[int] = None)-> int : #return the id of the fine-tuning model
        
        if (len(images) == 0) :
            raise Exception("'1-50 images required'")
        

        image_data = []
        for img in images :
            image_data.append(('tune[images][]', open(img ,'rb')))
            

        url = f"{self.url}tunes"
        myuuid = uuid.uuid4()
        uid = str(myuuid)

        data = {
            'tune[title]' : f"{description} - UUID - {uid}",
            'tune[name]' : name,
            'tune[token]' : token
        }
        if base_tune_id is not None :
            data['base_tune_id'] = base_tune_id

        response = requests.post(url, data=data, headers=self.headers, files=image_data)
        try :
            response = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)

        try :
            tuneid = response['id']
        except KeyError :
            raise ProviderException(message=response.text, code=response.status_code)

        return tuneid
    
    def image__generate_image(self, tuneid : int, prompt : str, negative_prompt : Optional[str] = "", num_images : Optional[int] = 1)->List[str] :
        url = f"{self.url}tunes/{tuneid}/prompts"
        data = {
            'prompt[text]' : prompt,
            'prompt[num_images]' : num_images,
            'prompt[negative_prompt]' : negative_prompt
        }

        response = requests.post(url, headers=self.headers, data=data)
        try : 
            response = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)

        promptid = response['id']

        list_url = self.wait_for_images(tuneid, promptid, num_images)

        return list_url
        

"""Method to create a fine tune model
    Create a tune with image__create_tune
    Create a prompt with image__create_prompt
    Request the output with image__get_image_prompt
"""
"""Output need to have
    id of the tune, id of the prompt, list of url of the image
"""



test = AstriaApi()
print(test.get__all_tunes())
#print(test.image__get_image_prompt(tuneid=950996, promptid=13182890))









