[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Eden AI Logo](EdenAI-WrittenLogo(1).png)

# EdenAI APIs

[Eden AI](https://www.edenai.co/?referral=github) aims to simplify the use and deployment of AI technologies by providing a unique API (application programming interface) that connects to the best possible AI engines. These engines are either proprietary or Open source AI engines, and can be used for different purposes, e.g, face
detection, OCR (receipt, invoice, table...), keyword extraction, sentiment analysis, face detection, and much more. These technologies are provided by the best suppliers in the market. We can cite briefly some of them: Amazon, Google, Microsoft, Dataleon, and Mindee and many others. Eden AI take care of providing to it’s clients the best AI engine suited to their projects, and this, with the goal of keeping AI light and easy to any developer. 

![EdenAI Gif](ezgif.com-gif-maker(1).gif)

## Package Installation

You can install the package with pip :

``` bash

    pip install https://github.com/edenai/edenai-apis 

```

## Quick Start

  Eden AI APIs is the open source package including necessary methods for using AI technologies from different AI providers (ex: google, amazon, clarifai .etc).

  You can list all availalbe (`provider`, `feature`, `subfeature`) using **list_features** method. The function will return a list of tuples. For example :

  ```python
    [
        ("google", "image", "object_detection"), 
        ("api4ai", "image", "anonymization"), 
        ("microsoft", "text", "keyword_extraction"), 
        ...
    ]

  ```

  You can use arguments to filter your request.

  ```python
    def list_features( provider_name: str = None, feature: str = None, subfeature: str = None, as_dict: bool = False) -> Union[List, Dict]:
  ```

  If you set as_dict to true you get a different formatting :

  ```python
      {
        [provider]:{
            [feature]: {
                [subfeature]: True
            }
        }
      }
  ```


  You can get a list of all providers given a feature and a subfeature using `list_providers`. If neither the feature or subfeature are passed within the function arguments, the function returns the list of all available providers.

  ```python
    def list_providers(feature: str = None, subfeature: str = None) -> List[str]
  ```


  Once you know what provider you want to run for what feature/subfeature you can execute the `compute_output` function

  ```python
    def compute_output(provider_name: str, feature: str, subfeature: str, args: Dict, phase: str = "", fake: bool = False) -> Dict
  ```

  You get either the result back or a `job_id` if the subfeature is asyncronous (ex: speech_to_text_asynx).

  If you're running an asyncronous feature (ex: speech to text, object extraction from videos ...etc ) then, when the computed subfeature using `compute_output` returns a `job_id`. Passing this `job_id` along a given provider, feature, subfeature and phase as arguments to `get_async_job_result` function returns the result of the asyncronous call.

  ```python
    def get_async_job_result(provider_name: str, feature: str, subfeature: str, async_job_id: str,
                                            phase: str = "", fake: bool = False, project_name: str = None) -> Dict:
  ```

## Contribute

We would love to have your contribution. Please follow our [gidelines for adding a new AI provider's API or a new AI feature](CONTRIBUTE.md). You can check [the package structure](PACKAGE_STRUCTURE.md) for more details on how it is organized.
We use GitHub issues for tracking requests and bugs. For broader discussions you can [join our discord](https://discord.com/invite/VYwTbMQc8u).

## Don’t want to create accounts for all providers and host the project by yourself?
You can create an account on [Eden AI](https://app.edenai.run/user/register?referral=github) and have access to all the AI technologies and providers directly through our API.
[![Eden AI Logo](EdenAI-WrittenLogo(1).png)](https://app.edenai.run/user/register?referral=github)


## Join the community!
Join our friendly community to improve your skills, focus on the integration of AI engines, get help to use Eden AI API and much more !

[![](https://dcbadge.vercel.app/api/server/VYwTbMQc8u)](https://discord.com/invite/VYwTbMQc8u)
[![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/edenai/) [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://edenai.medium.com/)

## License :
[Apache License 2.0](LICENSE)
