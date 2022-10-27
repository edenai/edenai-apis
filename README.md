![Eden AI Logo](EdenAI-WrittenLogo(1).png "Logo Title Text 1")

# EdenAI - Providers Connectors

Eden AI aims to simplify the use and deployment of AI technologies by providing a unique API (application programming interface) that connects to the best possible AI engines. These engines are either proprietary or Open source AI engines, and can be used for different purposes, e.g, face
detection, OCR (receipt, invoice, table...), keyword extraction, sentiment analysis, face detection, and much more. These technologies are provided by the best suppliers in the market. We can cite briefly some of them: Amazon, Google, Microsoft, Dataleon, and Mindee and many others. Eden AI take care of providing to it’s clients the best AI engine suited to their projects, and this, with the goal of keeping AI light and easy to any developer. 

[Join us on discord !](https://discord.com/invite/VYwTbMQc8u)

![EdenAI Gif](ezgif.com-gif-maker(1).gif)

## Introduction to the package

Eden AI's Providers Connectors is the open source package including necessary methods for using AI technologies from different [providers](#provider). The package contains principally `five packages` and `one interface module`:

### **interface**

The module interface is responsable of linking providers subfeature methods to our back-end project (which is, on the other hand, private). It containts principally five functions that you can use to interact with providers' subfeatures. We list the methods hereafter:

* #### list_features

  Returns possible combinations of  (`provider`, `feature`, `subfeature`) or (`provider`, `feature`, `subfeature`, `phase`) given a provider, feature and subfeature as a list if **as_dict** is set to False, otherwise returns the resutls as a dictionary. If neither the provider, feature or subfeature are passed withing the function arguments, it returns the list of all possible combinations.

  ```python
    def list_features( provider_name: str = None, feature: str = None, subfeature: str = None, as_dict: bool = False) -> Union[List, Dict]:
  ```

  *E.g of a dict results:*:

  ```python
      {
        [provider]:{
            [feature]: {
                [subfeature]: True
            }
        }
        or
        [provider]:{
            [feature]: {
                [subfeature]: {
                    [phase] : True
                }
            }
        }
      }
  ```

* #### list_providers

  Returns a list of providers given a feature and a subfeature: **List[str]**. If neither the feature or subfeature are passed within the function arguments, the function returns the list of all available providers.

  ```python
    def list_providers(feature: str = None, subfeature: str = None) -> List[str]
  ```

* #### compute_output

  Runs the actual computation of a triple (feature, subfeature, phase) for a specific provider. `Phase` can be not passed for arguments for subfeatures that do not require a phase (most of the subfeatures available in the project does not require a `phase`). The optional argument **fake** is set to `False` by default. When set to `True`, **compute_output** will return results from the sample output saved in the project.

  ```python
    def compute_output(provider_name: str, feature: str, subfeature: str, args: Dict, phase: str = "", fake: bool = False, user_email: str = None) -> Dict
  ```

* ### get_async_job_result

  When the computed subfeature using `compute_output` is **asyncronous**, a *`public_job_id`* is returned. Passing this *`public_job_id`* along a given provider, feature, subfeature and phase as arguments for the `get_async_job_result` function returns the result of the asyncronous call.

  ```python
    def get_async_job_result(provider_name: str, feature: str, subfeature: str, async_job_id: str,
                                            phase: str = "", fake: bool = False, project_name: str = None) -> Dict:
  ```

* ### check_provider_constraints

  check if a triple (provider, feature, subfeature)'s info constrains conforms to the given `constraints` dictionary argument

  ```python
    def check_provider_constraints(provider_name: str, feature: str, subfeature: str, phase: str = None, constraints: Dict = None) -> Tuple[bool, str]
  ```

## Notions

In the EdenAI organization, we follow a naming patterns based on `four` principle bricks: **provider**, **feature**, **subfeature** and **phase**. We explain each of them hereafter:

### **Provider**

`Provider` represents names of the suppliers from which we integrate our AI technologies. *E.g,: Amazon, Google, Microsoft, Dateleon and Mindee* and many others. To this time, *Eden AI* have integrated in it's platform more than 25 **providers**.

### **Subfeature**

`Subfeauture` represents the naming schema for the integrated `AI Technologies`. *E.g,: automatic_translation, text_to_speech, face_detection, invoice_parser or keyword_extraction*.

### **Feature**

To better apprehend the use of each of the `subfeatures` available in our platform, we grouped related subfeatures into categories that we named `features`. In other words, each feature encompasses a list of subfeatures. *E,g.:* the feature **audio** contains the two subfeatures: *speech_to_text_*[async](#async-logic) and *text_to_speech*.

### **Phase**

For some subfeatures, the computation may encompass different stages to execute, usually sequentially, but not always. These stages are named `phases`. *E.g,* the subfeature `seach` inside the feature `image` encompass four **phases**: *upload_image*, *get_image*, *get_images*, *launch_similarity*, *delete_image*.

### **Async logic**

Some subfeatures can be time consuming, like converting *speech* to *text*, and so, in a logic cycle of (request/response), the computing result can not be sent directly to the user. In this context, we propose an `Asynchrone logic`, in which the final result is differed, but instead, users can check reapedetly for the call result.

In order the support this implementation logic, subfeatures that are asynchrone ends with **_async** and are splitted into two methods call with two possible [`suffix`](#suffix): **__launch_job** and **__get_job_result**. *E.g,:*

```python
  def audio__speech_to_text_async__launch_job(self, file: BufferedReader, language: str) -> AsyncLaunchJobResponseType:
```

```python
  def audio__speech_to_text_async__get_job_result(self, provider_job_id: str) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
```

### **Suffix**

Used suffixes are usually either **__launch_job** or **__get_job_result**. The **__launch_job** suffix refers to the subfeature method responsable for starting the the asynchronous call. The **__get_job_result** refers to the subfeature method for handling the result of the asynchronous call.

### Standarized response

Responses to the same `subfeature` even if called from different providers are all `standarized` to a uniform response.

## Contribute

You are more than welcome to contribute to our project, just `Fork the project`, `push` your modifications into a `branch` and open a `Pull Request`. We will be happy to check and eventually merge your **contributions** into our project.

## Join the community!
Join our friendly community to improve your skills, focus on the integration of AI engines, get help to use Eden AI API and much more !
[![Discord Logo](Discord-Logo (1).png)](https://discord.com/invite/VYwTbMQc8u)

## License :
[Apache License 2.0](LICENSE)
