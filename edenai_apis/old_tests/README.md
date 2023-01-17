
# Tests

### test_classes.py
Test if providers classes are well formed :

- Inherit from ProviderInterface
- implement a well formatted info.json containing subfeatures versions
- implement all features defined in info.json

### test_features_auto.py
Test all subfeatures (except image search) for all providers to check if:
- Saved output for each provider exists and is well standardized
- providers APIs work and their outputs are well standardized

### test_image_search.py
Test image search subfeature.
The tests are handled differently because the subfeature implements phases

### test_interface.py
Test interface functions:
- compute_output
- list_features 
- list_providers
- check_provider_constraints

### test_language.py
Test language standardization functions.
Different Providers handle different language formats.
We implement language utils to handle the standardisation.

### test_loaders.py
Test load_feature, load_provider functions.
