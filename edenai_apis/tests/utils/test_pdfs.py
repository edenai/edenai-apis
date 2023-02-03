import pytest

from edenai_apis.utils.pdfs import get_pdf_width_height

class TestGetPdfWidthHeight:
    def test_valid_pdf_no_error_and_check_return_type(self):
        with open('edenai_apis/features/ocr/data/resume.pdf', 'rb') as f:
            (width, height) = get_pdf_width_height(f)

            assert isinstance(width, float)
            assert isinstance(height, float)