import os

import pytest
from settings import base_path

from edenai_apis.utils.upload_s3 import (
    get_providers_json_from_s3,
    s3_client_load,
    upload_file_to_s3,
)


@pytest.mark.e2e
def test_s3_client_load():
    client = s3_client_load()
    assert client != None


@pytest.mark.e2e
def test_upload_to_s3():
    file_path = os.path.join(base_path, "features/ocr/data/resume.pdf")
    response = upload_file_to_s3(file_path, "test.pdf")
    assert response != None


@pytest.mark.e2e
def test_get_providers_json_from_s3():
    providers_info = get_providers_json_from_s3()
    assert isinstance(providers_info, dict)
