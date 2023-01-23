import pytest

from edenai_apis.utils.upload_s3 import (
    get_providers_json_from_s3,
    s3_client_load,
    upload_file_to_s3
)

def test_s3_client_load():
    client = s3_client_load()
    assert client != None

def test_upload_to_s3():
    with open('edenai_apis/features/ocr/data/resume.pdf', 'rb') as f:
        response = upload_file_to_s3(f, 'test.pdf')
        assert response != None

def test_get_providers_json_from_s3():
    providers_info = get_providers_json_from_s3()
    assert isinstance(providers_info, dict)