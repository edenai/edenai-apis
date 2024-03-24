import base64
import json

import pytest
import requests
import params
class TestVisitURLForm:
    @pytest.fixture(scope="class")

    def test(self):

        assert True
    def test_submit_form_with_base64_pdf(self, test):

        self.api_key = params.hirize_api_key
        self.url = "https://connect.hirize.hr/api/public/parser?api_key=" + self.api_key
        self.headers = {
                            'Content-Type': 'application/json'
                       }
        dumpData = json.dumps({
            "payload": params.file_payload,
            "file_name": params.file_name
        })

        hirize_response =  requests.request("POST", self.url, headers=self.headers, data=dumpData)

        assert hirize_response.status_code == 201
