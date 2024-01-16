from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.utils.types import ResponseType

from edenai_apis.features.image.fine_tuning import (
    FineTuningCreateProjectDataClass,
    FineTuningGenerateImageDataClass,
    FineTuningListProject
)

from edenai_apis.utils.types import AsyncLaunchJobResponseType, AsyncPendingResponseType

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

    def image__finetuning_wait_for_images(self, tuneid :int, promptid :int) ->ResponseType[FineTuningGenerateImageDataClass]:
        lst_url = self.image__finetuning_get_image_prompt(tuneid, promptid)
        
        if lst_url != [] :
            return lst_url
        
        return lst_url

    def image__finetuning_promptinfo(self, tuneid : int, promptid : int)->json :
        '''Get all the information about a specific prompt'''
        response = requests.get(f"{self.url}tunes/{tuneid}/prompts/{promptid}", headers=self.headers)
        if response.status_code == 200 :
            try :
                resjson = response.json()
            except json.JSONDecodeError :
                raise ProviderException(message=response.text, code=response.status_code)
            
            return resjson
        else :
            raise ProviderException(message=response.text, code=response.status_code)
    
    def image__finetuning_get_image_prompt(self, tuneid : int, promptid : int) ->ResponseType[FineTuningGenerateImageDataClass]:
        promptinfo = self.image__finetuning_promptinfo(tuneid, promptid)
        try :
            images = promptinfo['images']
        except KeyError :
            raise ProviderException(message=promptinfo)
        
        return ResponseType[FineTuningGenerateImageDataClass](
            original_response=promptinfo,
            standardized_response=images
        )

    def image__finetuning_get__all_tunes(self) ->ResponseType[FineTuningListProject]: 
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
            base_project_id : Optional[int] = None
            ) -> AsyncLaunchJobResponseType : 
        
        if (len(images) == 0) :
            raise ProviderException("'1-50 images required'")
        

        image_data = []
        for img in images :
            image_data.append(('tune[images][]', open(img ,'rb')))
            

        url = f"{self.url}tunes"
        myuuid = uuid.uuid4()
        uid = str(myuuid)

        description = f"{description} - UUID - {uid}"
        data = {
            'tune[title]' : description,
            'tune[name]' : name,
        }
        if base_project_id is not None :
            data['base_tune_id'] = base_project_id

        response = requests.post(url, data=data, headers=self.headers, files=image_data)
        try :
            resjson = response.json()
        except json.JSONDecodeError :
            raise ProviderException(message=response.text, code=response.status_code)

        try :
            tuneid = response['id']
        except KeyError :
            raise ProviderException(message=response.text, code=response.status_code)
        

        return AsyncLaunchJobResponseType(provider_job_id=tuneid)
    
    def image__finetuning__create_project_async__get_job_result(
            self, 
            job_id : str) ->AsyncPendingResponseType: 
        
        response = requests.get(f'https://api.astria.ai/tunes/{job_id}', headers=self.headers)
        if response.status_code == 200 :
            try :
                resjson = json.loads(response)
            except json.JSONDecodeError :
                raise ProviderException(message=response.text, code=response.status_code)

            result = resjson['trained_at']

            if resjson['trained_at'] != None :
                return AsyncPendingResponseType(status=f'trained_at : {result}')
            else :
                return AsyncPendingResponseType(status="Pending")
        
        else :
            raise ProviderException(message=response.text, code=response.status_code)


    
    def image__finetuning__generate_image_async__launch_job(
            self, 
            tuneid : int, 
            prompt : str, 
            negative_prompt : Optional[str] = "", 
            num_images : Optional[int] = 1)->AsyncLaunchJobResponseType: 
        
        
        url = f"{self.url}tunes/{tuneid}/prompts"
        data = {
            'prompt[text]' : prompt,
            'prompt[num_images]' : num_images,
            'prompt[negative_prompt]' : negative_prompt
        }

        response = requests.post(url, headers=self.headers, data=data)
        if response.status_code == 200 :
            try : 
                rjson = response.json()
            except json.JSONDecodeError :
                raise ProviderException(message=response.text, code=response.status_code)
            
            try :
                promptid = rjson['id']
            except KeyError :
                raise ProviderException(message=response, code=response.status_code)

            tune_prompt_id = f'{tuneid}-{promptid}'

            return AsyncLaunchJobResponseType(provider_job_id=tune_prompt_id)
        else :
            raise ProviderException(response = response.text, code = response.status_code)
    
    
    def image__finetuning__async_generate_image__get_job_result(
            self, 
            provider_job_id: str)->ResponseType[FineTuningGenerateImageDataClass] : 
        
        tune_prompt = provider_job_id.split('-')
        tuneid = tune_prompt[0]
        promptid = tune_prompt[1]

        output = self.image__finetuning_wait_for_images(tuneid, promptid) #Good class already

        return output 



    def image__finetuning__list_project(self)->ResponseType[FineTuningListProject]:
        response = self.image__finetuning_get__all_tunes()
        
        lst_id_tunes = []
        for tunes in response :
            lst_id_tunes.append(tunes['id'])

        return ResponseType[FineTuningListProject](
            original_response=response.text,
            standardized_response=lst_id_tunes
        )

        

"""Method to create a fine tune model
    Create a tune with image__finetuning__create_project_async__launch_job
    Look if the model has finished to train with image__finetuning__async_generate_image__get_job_result
    Create a prompt the images with image__finetuning__generate_image_async__launch_job
    Request the output of url images with image__finetuning__async_generate_image__get_job_result

    You can recover all of the projects id with image__finetuning__list_project
"""




