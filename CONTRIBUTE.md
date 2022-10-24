
# Table of Contents
0.  [Package structure presentation](#org02bb1ec)
1.  [Detailed explanation](#org02ff1ec)
    1.  [Create a folder with you provider name under `providers/apis` directory.](#orga3ef4f7)
    2.  [Add a sample directory for the feature and add corresponding files](#org97d5614)
    3.  [Tests](#org3fd3a19)
2.  [Checklist](#org0993329)

<a id="org02bb1ec"></a>

# Package structure presentation

### **api**

The `api` package is grouped by providers name, each of this providers folder containing:

* `provider_`**api**`.py` : The module containing all the provider's AI technologies `methods` listed inside the provider `class` under the following format: `feature`\_\_`subfeature`\_\_`phase`\_\_`sufix`, where `phase` and `suffix` can be nulls. *E,g*, the ``` text__sentiment_analysis() ``` *method* inside the `AmazonApi` *class* in the `amazon_api.py` module, where **text** represents the [feature](#feature) and **sentiment_analysis** the [subfeature](#subfeature) or the AI technologie you are going to use. *E.g,:*

  ```python
    def image__explicit_content(
        self, file: BufferedReader
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
* `subfeature_`**response**`.json` : the subfeature [standarized response](#standarized-response) resulting from mapping the the call result to the appropriate *dataclass*.

### **loaders**

The `loaders` package is used to load data either from from the `providers` package or from the `features` one. It contains two functions, each one responsible for loading data for each of the two previous packages. *E.g:*

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


<a id="org02ff1ec"></a>

# Detailed explanation


<a id="orga3ef4f7"></a>

## Create a folder with you provider name under `providers/apis` directory.

Directory should look like this:

    providers/apis/<provider>
    ├── <provider>_api.py
    ├── config.py (optional)
    ├── info.json
    ├── __init__.py
    └── outputs
        └── <feature>
            └── <subfeature>_output.json

folder should contain:

-   `config.py` *optional*
-   `info.json`
    a json file wich indicate subfeatures infos for each features
    eg:
    
        {
            "text": {
                "summarize": {
                    "version": "v1.0"
                    "constraints": { # this is optional
                       "languages": [
                           'en',
                           'fr',
                           ...
                       ] 
                    }
                }
            }
        }
-   the main file `<provider-name>_api.py`
    should contain a class with your provider name that will inherit from `ProviderApi` class.
    it will also inherit from other abstract classes representing each features
    You will define feature methods according to the features class abstractmethods.
    each feature method has to return a Dictionary:
    containing the original response of the provider and your standarized response
    
    eg:
    
        class AmazonApi(
            ProviderApi,
            Image
            Ocr,
        ):
            def image__object_detection(file_name: str, file_content: io.BytesIO) -> Dict:
                """your code here"""
                return {"original_response": response, "standarized_response": standarized}

-   An output directory containing one directory by feature, each feature directory will contain output json files representing original response returned by provider for a subfeature


<a id="org97d5614"></a>

## Add a sample directory for the feature and add corresponding files

    providers/features/<feature>
    ├── __init__.py
    ├── <feature>_class.py
    └── <subfeature>
        ├── <subfeature>_args.py
        ├── <subfeature>_dataclass.py
        └── <subfeature>_response.json

-   \<feature>_class.py abstract class for method
-   \<subfeature>_args.py to generate arguments for tests
-   \<subfeature>_dataclass.py class for output
-   \<subfeature>_response.json: validated standarized response to compare with your own standarization


<a id="org3fd3a19"></a>

## Tests

**Tests should be run at the root of your project**

-   Generate an ouptut from your feature method:
    
        pytest -s providers/tests/utils/outputs.py --provider <provider> --feature <feature> --subfeature <subfeature>
-   Run all projects tests:
    
        pytest
-   Test classes only (without api call) 
    
        pytest providers/tests/test_classes.py
-   Test one provider with all subfeatures (example with microsoft) 
    
        pytest -m microsoft
-   Test all provider with one subfeature (example with invoice_parser) 
    
        pytest -m invoice_parser
-   Test with "and" or "or"
    
        pytest -m "microsoft and invoice_parser"
    
        pytest -m "resume_parser or invoice_parser"

**Usefull options**
1. `-s` : show output (print)
1. `-k test_method` : execute only test_method
1. `-n auto` : execute test in parallel
1. `-x` : stop at first fail

<a id="org0993329"></a>

