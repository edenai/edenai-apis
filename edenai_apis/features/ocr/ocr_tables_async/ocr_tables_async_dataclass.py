from typing import List, Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class BoundixBoxOCRTable(BaseModel):
    left: float = 0
    top: float = 0
    width: float = 0
    height: float = 0


class Cell(BaseModel):
    text: Optional[StrictStr]
    row_span: int
    col_span: int
    row_index: int
    col_index: int
    confidence: Optional[float]
    is_header: bool = False
    bounding_box: BoundixBoxOCRTable = BoundixBoxOCRTable()


class Row(BaseModel):
    cells: List[Cell] = Field(default_factory=list)


class Table(BaseModel):
    rows: List[Row] = Field(default_factory=list)
    num_rows: int
    num_cols: int


class Page(BaseModel):
    tables: List[Table] = Field(default_factory=list)


class OcrTablesAsyncDataClass(BaseModel):
    pages: List[Page] = Field(default_factory=list)
    num_pages: Optional[int]
