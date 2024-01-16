from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.utils.types import ResponseType


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

    def image__finetuning_wait_for_images(self, tuneid :int, promptid :int) ->List[str]:
        lst_url = self.image__finetuning_get_image_prompt(tuneid, promptid)
        
        if lst_url != [] :
            return lst_url
        
        return lst_url

    def image__finetuning_promptinfo(self, tuneid : int, promptid : int)->json :
        '''Get all the information about a specific prompt'''
        response = requests.get(f"{self.url}tunes/{tuneid}/prompts/{promptid}", headers=self.headers)
        try :
            response = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)
        return response
    
    def image__finetuning_get_image_prompt(self, tuneid : int, promptid : int) ->List[str]:
        promptinfo = self.image__finetuning_promptinfo(tuneid, promptid)
        try :
            images = promptinfo['images']
        except KeyError :
            raise ProviderException(message=promptinfo)
        return images

    def image__finetuning_get__all_tunes(self) ->json: #ResponseType[FineTuningListProjectDataClass]
        response = requests.get(f'{self.url}tunes', headers=self.headers)
        try : 
            response = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)
        return response
    
    def image__finetuning_get_tune(self, tuneid : int = None)->json :
        if tuneid is not None :
            response = requests.get(f"{self.url}tunes/{tuneid}", headers=self.headers)
            try : 
                response = response.json()
            except json.JSONDecodeError :
                raise ProviderException(message=response.text, code=response.status_code)
            return response
        raise ProviderException("Missing id")


    def image__finetuning__create_project_async__launch_job(
            self, 
            name : str, 
            description : str, 
            images : List[str], 
            token : Literal['owhx', 'sks'] = 'sks', 
            base_tune_id : Optional[int] = None) -> int : #ResponseType[FineTuningCreateProjectDataClass]
        
        if (len(images) == 0) :
            raise ProviderException("'1-50 images required'")
        

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

        return tuneid #change class
    
    def image__finetuning__create_project_async__get_job_result(self, job_id : str) : #ResponseType[Job done]
        response = requests.get(f'https://api.astria.ai/tunes/{job_id}', headers=self.headers)
        if response.status_code == 200 :
            try :
                response = json.loads(response)
            except json.JSONDecodeError :
                raise ProviderException(message=response.text, code=response.status_code)

            if response['trained_at'] != None :
                return #AsynResponstype, job done
            else :
                return #pending async (look amazon)
        
        else :
            raise ProviderException(message=response.text, code=response.status_code)


    
    def image__finetuning__generate_image_async__launch_job(
            self, 
            tuneid : int, 
            prompt : str, 
            negative_prompt : Optional[str] = "", 
            num_images : Optional[int] = 1)->List[str] : #ResponseType[FineTuningGenerateImageDataClass]
        
        
        url = f"{self.url}tunes/{tuneid}/prompts"
        data = {
            'prompt[text]' : prompt,
            'prompt[num_images]' : num_images,
            'prompt[negative_prompt]' : negative_prompt
        }

        response = requests.post(url, headers=self.headers, data=data)
        try : 
            rjson = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)
        
        print(response)
        try :
            promptid = rjson['id']
        except KeyError :
            raise ProviderException(message=response, code=response.status_code)
        
        return f'{tuneid}-{promptid}'
        
    
    
    def image__finetuning__async_generate_image__get_job_result(self, provider_job_id: str) : #ResponseType[All urls]
        tune_prompt = provider_job_id.split('-')
        tuneid = tune_prompt[0]
        promptid = tune_prompt[1]

        list_url = self.image__finetuning_wait_for_images(tuneid, promptid)

        return list_url #change output



    def image__finetuning__list_tunes(self):
        response = self.image__finetuning_get__all_tunes()
        
        lst_id_tunes = []
        for tunes in response :
            lst_id_tunes.append(tunes['id'])

        return lst_id_tunes

        

"""Method to create a fine tune model
    Create a tune with image__create_tune
    Create a prompt and request the images with image__generate_image 

Output :
    Current output : list of url of the images
    Will be good to integrate in the output the promptid and the tuneid
"""
test = AstriaApi()
res = test.image__finetuning__generate_image_async__launch_job(956910, "A little sks dog in a bed", num_images=4)
print(res)
time.sleep(120)
img = test.image__finetuning__async_generate_image__get_job_result(res)
print(img)

#add class and finish implémentation




