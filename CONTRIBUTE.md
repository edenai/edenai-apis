
# Table of Contents
1.  [Add a new provider](#orga3ef4f7)
2.  [Add a new feature](#org97d5614)
3.  [Add your secret keys](#org8h7as8i)
4.  [Tests](#org3fd3a19)


<a id="orga3ef4f7"></a>

## Create a folder with you provider name under `edenai_apis/apis` directory.

Directory should look like this:

    edenai_apis/apis/<provider>
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
    should contain a class with your provider name that will inherit from `ProviderInterface` class.
    it will also inherit from other abstract classes representing each features
    You will define feature methods according to the features class abstractmethods.
    each feature method has to return a Dictionary:
    containing the original response of the provider and your standardized response
    
    eg:
    
        class AmazonApi(
            ProviderInterface,
            Image
            Ocr,
        ):
            def image__object_detection(self, file: str, model: str= None, file_url: str = "") -> Dict:
                """your code here"""
                return {"original_response": response, "standardized_response": standardized}

    Within your feature method, your api or sdk integration may throw and exception or return an error. In this case, you should raise a specific exception: a `ProviderException` with an *exception message* and an `optional` *error code*.

    eg:

        def image__object_detection(self, file: str, model: str=None, file_url: str = "") -> Dict:
            """your code here"""
            ...
            if error_condition: # if status code equal to 400 for example
                raise ProviderException('exception message', status_code)
            ...
            return {"original_response": response, "standardized_response": standardized}


-   An output directory containing one directory by feature, each feature directory will contain output json files representing original response returned by provider for a subfeature


<a id="org97d5614"></a>

## Add a sample directory for the feature and add corresponding files

    edenai_apis/features/<feature>
    ├── __init__.py
    ├── <feature>_class.py
    └── <subfeature>
        ├── <subfeature>_args.py
        ├── <subfeature>_dataclass.py
        └── <subfeature>_response.json

-   \<feature>_class.py abstract class for method
-   \<subfeature>_args.py to generate arguments for tests
-   \<subfeature>_dataclass.py class for output
-   \<subfeature>_response.json: validated standardized response to compare with your own standarization


<a id="org8h7as8i"></a>

## Add a settings JSON file which will contain your personal key/secret keys

    edenai_apis
    ├── __init__.py
    ├── apis
    ├── ...
    └── api_keys
        ├── amazon_settings.json
        ├── microsoft_settings.json
        ├──...
        └── <provider>_settings.json


your settings file should be named following this pattern: provider`_settings`.json, and should contains a json object with key/values paires.

eg:

    {
        "key": "key_value" 
    }

<a id="org3fd3a19"></a>

## Tests

**Tests should be run at the root of your project**

-   Generate an ouptut from your feature method:
    
        pytest -s edenai_apis/tests/outputs.py --provider <provider> --feature <feature> --subfeature <subfeature>
-   Run all projects tests:
    
        pytest
-   Test classes only (without api call) 
    
        pytest edenai_apis/tests/test_classes.py
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

