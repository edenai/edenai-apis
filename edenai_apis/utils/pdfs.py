from io import BufferedReader
from typing import Tuple

import pypdf


def get_pdf_width_height(pdf_file: BufferedReader) -> Tuple[float, float]:
    """
    Read a pdf file and returns its width and height

    Args:
        - pdf_file (io.BufferedReaer): a pdf file

    Returns:
        - width, height: a tuple(float, float) representing width & height
    """
    reader = pypdf.PdfReader(pdf_file)
    rectangle_box = reader.pages[0].mediabox
    width = float(rectangle_box.width)
    height = float(rectangle_box.height)

    return width, height
