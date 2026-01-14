# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eden AI APIs is an open-source library that provides a unified interface to multiple AI providers (Google, Amazon, Microsoft, OpenAI, etc.). It standardizes API calls across 70+ providers for features like text analysis, image recognition, OCR, translation, audio processing, and LLMs.

## Common Commands

### Running Tests

```bash
# Run all tests
pytest

# Run tests for a specific provider
pytest -m microsoft

# Run tests for a specific subfeature
pytest -m invoice_parser

# Run tests with combined filters
pytest -m "microsoft and invoice_parser"
pytest -m "resume_parser or invoice_parser"

# Run class structure tests only (no API calls)
pytest edenai_apis/tests/test_classes.py

# Generate output from a provider's feature method
pytest -s edenai_apis/tests/outputs.py --provider <provider> --feature <feature> --subfeature <subfeature>

# Useful pytest options
# -s : show print output
# -k test_method : run specific test method
# -n auto : run tests in parallel
# -x : stop at first failure
```

### Type Checking and Linting

```bash
mypy edenai_apis/
pylint edenai_apis/
```

## Architecture

### Naming Convention: Feature/Subfeature/Phase

The codebase uses a hierarchical naming pattern:

- **Feature**: Category of AI capability (text, image, ocr, audio, video, translation, multimodal, llm)
- **Subfeature**: Specific AI technology (e.g., `sentiment_analysis`, `object_detection`, `invoice_parser`)
- **Phase**: Optional stage for multi-step operations (e.g., `upload_image`, `launch_similarity`)

Method names follow the pattern: `feature__subfeature__phase__suffix`
Example: `image__search__upload_image`, `audio__speech_to_text_async__launch_job`

### Core Package Structure

```
edenai_apis/
├── apis/                    # Provider implementations (one folder per provider)
│   └── <provider>/
│       ├── <provider>_api.py      # Main API class inheriting from ProviderInterface
│       ├── <provider>_<feature>_api.py  # Feature-specific mixin classes
│       ├── info.json              # Provider capabilities and constraints
│       ├── config.py              # Client configuration (optional)
│       └── outputs/               # Sample API responses for testing
├── features/                # Feature definitions and interfaces
│   └── <feature>/
│       ├── <feature>_interface.py  # Abstract interface with method signatures
│       └── <subfeature>/
│           ├── <subfeature>_dataclass.py  # Standardized response dataclass
│           └── <subfeature>_args.py       # Test arguments
├── interface.py             # Main compute_output() and helper functions
├── interface_v2.py          # Simplified interface (Text, Image, Audio, etc.)
├── loaders/                 # Data loading utilities for providers and features
└── utils/                   # Shared utilities (exceptions, file handling, etc.)
```

### Implementing a Provider API

Provider classes use multiple inheritance to compose features:

```python
class AmazonApi(
    ProviderInterface,
    AmazonOcrApi,
    AmazonAudioApi,
    AmazonImageApi,
    ...
):
    provider_name = "amazon"
```

Each feature method must return a dict with `original_response` and `standardized_response`:

```python
def image__object_detection(self, file: str, ...) -> ResponseType[ObjectDetectionDataClass]:
    # Call provider API
    response = self.client.detect_objects(...)
    # Standardize response
    standardized = ObjectDetectionDataClass(items=[...])
    return ResponseType(original_response=response, standardized_response=standardized)
```

### Response Types

- `ResponseType[T]`: Standard synchronous response with `original_response` and `standardized_response`
- `AsyncLaunchJobResponseType`: Returns `provider_job_id` for async operations
- `AsyncResponseType[T]`: Async result with `status` ("pending", "succeeded", "failed")

### Error Handling

Raise `ProviderException` (or subclasses) for provider errors:

```python
from edenai_apis.utils.exception import ProviderException, ProviderInvalidInputError

if error_condition:
    raise ProviderException("error message", code=400)
```

Exception hierarchy includes: `ProviderAuthorizationError`, `ProviderTimeoutError`, `ProviderInvalidInputFileError`, `ProviderParsingError`, etc.

### Async Operations

For time-consuming operations (speech-to-text, video analysis), use the async pattern with `__launch_job` and `__get_job_result` suffixes:

```python
def audio__speech_to_text_async__launch_job(self, file, ...) -> AsyncLaunchJobResponseType:
    # Start job, return provider_job_id

def audio__speech_to_text_async__get_job_result(self, provider_job_id) -> AsyncBaseResponseType[...]:
    # Check status and return result when ready
```

### API Keys Configuration

Provider credentials go in `edenai_apis/api_keys/<provider>_settings.json`. Copy from the template file `<provider>_settings_template.json` and add your keys.
