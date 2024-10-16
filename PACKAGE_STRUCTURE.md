# Package structure presentation

## Introduction to the package

Eden AI's Providers Connectors is the open source package including necessary methods for using AI technologies from different [providers](#provider). The package contains principally `five packages` and `one interface module`:

### **interface**

The module interface is responsible of linking providers subfeature methods to our back-end project (which is, on the other hand, private). It contains principally five functions that you can use to interact with providers' subfeatures. We list the methods hereafter:

* #### list_features

  Returns possible combinations of  (`provider`, `feature`, `subfeature`) or (`provider`, `feature`, `subfeature`, `phase`) given a provider, feature and subfeature as a list if **as_dict** is set to False, otherwise returns the results as a dictionary. If neither the provider, feature or subfeature are passed within the function arguments, it returns the list of all possible combinations.

  ```python
    def list_features( provider_name: str = None, feature: str = None, subfeature: str = None, as_dict: bool = False) -> Union[List, Dict]:
  ```

  *E.g. of a dict results:*

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

  When the computed subfeature using `compute_output` is **asynchronous**, a *`public_job_id`* is returned. Passing this *`public_job_id`* along a given provider, feature, subfeature and phase as arguments for the `get_async_job_result` function returns the result of the asyncronous call.

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

`Provider` represents names of the suppliers from which we integrate our AI technologies. *E.g.: Amazon, Google, Microsoft, Dateleon and Mindee* and many others. To this time, *Eden AI* has integrated in it's platform more than 25 **providers**.

### **Subfeature**

`Subfeature` represents the naming schema for the integrated `AI Technologies`. *E.g,: automatic_translation, text_to_speech, face_detection, invoice_parser or keyword_extraction*.

### **Feature**

To better apprehend the use of each of the `subfeatures` available in our platform, we grouped related subfeatures into categories that we named `features`. In other words, each feature encompasses a list of subfeatures. *E.g.:* the feature **audio** contains the two subfeatures: *speech_to_text_*[async](#async-logic) and *text_to_speech*.

### **Phase**

For some subfeatures, the computation may encompass different stages to execute, usually sequentially, but not always. These stages are named `phases`. *E.g.* the subfeature `seach` inside the feature `image` encompass four **phases**: *upload_image*, *get_image*, *get_images*, *launch_similarity*, *delete_image*.

```python
  def image__search__create_project(self, project_name: str) -> str:
    ...
    return project_id 
```

```python
  def image__search__upload_image(self, file: str, image_name: str, project_id: str, file_url: str = "") -> ResponseSuccess:
```

```python
  def image__search__delete_image(self, image_name: str, project_id: str)-> ResponseSuccess:
```

```python
  def image__search__get_image(self, image_name: str, project_id: str) -> ResponseType[SearchGetImageDataClass]:
```

In this case for example, in order retreive an image using the `image__search__get_image` method, you must first create a project with calling the method `image__search__create_project`. After that, you should upload some images using the `image__search__upload_image` method.

### **Async logic**

Some subfeatures can be time consuming, like converting *speech* to *text*, and so, in a logic cycle of (request/response), the computing result can not be sent directly to the user. In this context, we propose an `Asynchrone logic`, in which the final result is differed, but instead, users can check repeatedly for the call result.

In order to support this implementation logic, subfeatures that are asynchronous end with **_async** and are splitted into two methods called with two possible [`suffix`](#suffix): **__launch_job** and **__get_job_result**. *E.g.:*

```python
  def audio__speech_to_text_async__launch_job(self, file: str, language: str, ...) -> AsyncLaunchJobResponseType:
```

```python
  def audio__speech_to_text_async__get_job_result(self, provider_job_id: str) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
```

### **Suffix**

Used suffixes are usually either **__launch_job** or **__get_job_result**. The **__launch_job** suffix refers to the subfeature method responsible for starting the the asynchronous call. The **__get_job_result** refers to the subfeature method for handling the result of the asynchronous call.

### Standardized response

Responses to the same `subfeature` even if called from different providers are all `standardized` to a uniform response.

### Language Standardization

Language standardization is the process in which language constraints are managed within our system. For more information, please refer to the following [link of our documentation](https://docs.edenai.co/docs/language-standardization). 


### **api**

The `api` package is grouped by providers name, each of this providers folder containing:

* `provider_`**api**`.py` : The module containing all the provider's AI technologies `methods` listed inside the provider `class` under the following format: `feature`\_\_`subfeature`\_\_`phase`\_\_`sufix`, where `phase` and `suffix` can be nulls. *E,g*, the ``` text__sentiment_analysis() ``` *method* inside the `AmazonApi` *class* in the `amazon_api.py` module, where **text** represents the [feature](#feature) and **sentiment_analysis** the [subfeature](#subfeature) or the AI technologie you are going to use. *E.g,:*

  ```python
    def image__explicit_content(
        self, file: str, file_url: str = ""
    ) -> ResponseType[ExplicitContentDataClass]:
        file_content = file.read()
        response = clients["image"].detect_moderation_labels(
            Image={"Bytes": file_content}, MinConfidence=20
        )
        ........
        ......
  ```

* `info.json` : The file containing **json** informations about the available [subfeatures](#subfeature) for the provider package.

* `helpers \& confif files` : Additional files needed to process correctly each of the [subfeatures](#subfeature).

### **features**

The package listing all the [features](#feature) available in the project, each of the feature package containing a **data** folder containing raw data for testing the subfeatures + a list of [subfeatures](#subfeature) associated to the feature. Each subfeature package containing:

* `subfeature_`**args**`.py` : the subfeature input **args** or parameters used for testing.
* `subfeature_`**dataclass**`.py` : the subfeature **dataclass** used for mapping the subfeature call result to a uniform dictionary (json).
* `subfeature_`**response**`.json` : the subfeature [standardized response](#standardized-response) resulting from mapping the the call result to the appropriate *dataclass*.

### **loaders**

The `loaders` package is used to load data either from the `providers` package or from the `features` one. It contains two functions, each one responsible for loading data for each of the two previous packages. *E.g:*

  ```python

    class ProviderDataEnum(Enum):
      INFO_FILE = "load_info_file"
      CLASS = "load_class"
      OUTPUT = "load_output"
      SUBFEATURE = "load_subfeature"
      PROVIDER_INFO = "load_provider_subfeature_info"
      KEY = "load_key"

    def load_provider( data_provider: ProviderDataEnum,
      provider_name: str = "",
      feature: str = "",
      subfeature: str = "",
      phase: str = "",
      suffix: str = "",
      **kwargs):

    translation_output = load_provider(ProviderDataEnum.OUTPUT, "amazon" , "translation", "automatic_translation")

  ```

### **tests**

The `tests` package groupes all tests used in *`EdenAI - Providers Connectors`*. For more details about tests, please refer to the following [README.md file](providers/tests/README.md)

### **utils**

The package `utils` encompass different util functions used to aid subfeatures' computation.


