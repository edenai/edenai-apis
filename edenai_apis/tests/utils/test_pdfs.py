import os

import pytest
from settings import base_path

from edenai_apis.utils.pdfs import get_pdf_width_height


class TestGetPdfWidthHeight:
    @pytest.mark.integration
    def test_valid_pdf_no_error_and_check_return_type(self):
        file_pdf_path = os.path.join(base_path, "features/ocr/data/resume.pdf")
        with open(file_pdf_path, "rb") as f:
            (width, height) = get_pdf_width_height(f)

            assert isinstance(width, float)
            assert isinstance(height, float)
