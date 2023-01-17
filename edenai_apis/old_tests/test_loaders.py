"""
    Test load_feature, load_provider functions.
"""
import os
from typing import Callable
import pytest

from pydantic import BaseModel
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.apis.amazon.amazon_api import AmazonApi


VALID_PROVIDER = "amazon"
VALID_FEATURE = "audio"
VALID_SUBFEATURE = "text_to_speech"
def test_load_output():
    output = load_provider(
        ProviderDataEnum.OUTPUT, VALID_PROVIDER, VALID_FEATURE, VALID_SUBFEATURE
    )
    assert "standardized_response" in output and "original_response" in output


@pytest.mark.skipif(os.environ.get("TEST_SCOPE") == 'CICD-OPENSOURCE', reason="Don't run on opensource cicd workflow")
def test_load_key():
    # without location
    key = load_provider(ProviderDataEnum.KEY, VALID_PROVIDER)
    assert isinstance(key, dict)

    # with weird argument location
    key = load_provider(ProviderDataEnum.KEY, VALID_PROVIDER, locat=True)
    assert isinstance(key, dict)

    # with argument location to True
    key = load_provider(ProviderDataEnum.KEY, VALID_PROVIDER, location=True)
    assert (
        isinstance(key, tuple) and isinstance(key[0], dict) and isinstance(key[1], str)
    )


def test_load_class():
    class_instance = load_provider(ProviderDataEnum.CLASS, VALID_PROVIDER)
    assert class_instance == AmazonApi


@pytest.mark.skipif(os.environ.get("TEST_SCOPE") == 'CICD-OPENSOURCE', reason="Don't run on opensource cicd workflow")
def test_load_subfeature():
    subfeature = load_provider(
        ProviderDataEnum.SUBFEATURE, VALID_PROVIDER, VALID_FEATURE, VALID_SUBFEATURE
    )
    assert isinstance(subfeature, Callable)


def test_load_dataclass():
    data_class = load_feature(
        FeatureDataEnum.DATA_CLASS, feature=VALID_FEATURE, subfeature=VALID_SUBFEATURE
    )
    assert issubclass(data_class, BaseModel)


def test_load_samples():
    samples = load_feature(
        FeatureDataEnum.SAMPLES_ARGS, feature=VALID_FEATURE, subfeature=VALID_SUBFEATURE
    )
    assert isinstance(samples, dict)


def test_load_provider_subfeature_info():
    provider_subfeature_info = load_provider(
        ProviderDataEnum.PROVIDER_INFO,
        VALID_PROVIDER,
        VALID_FEATURE,
        VALID_SUBFEATURE,
    )
    assert (
        isinstance(provider_subfeature_info, dict)
        and "constraints" in provider_subfeature_info
    )
