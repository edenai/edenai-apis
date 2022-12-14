from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class BoundixBoxOCRTable(BaseModel):
    left: Optional[float]
    top: Optional[float]
    width: Optional[float]
    height: Optional[float]


class Cell(BaseModel):
    text: Optional[StrictStr]
    row_span: Optional[int]
    col_span: Optional[int]
    confidence: Optional[float]
    bounding_box: BoundixBoxOCRTable = BoundixBoxOCRTable()


class Row(BaseModel):
    cells: Sequence[Cell] = Field(default_factory=list)
    is_header: bool = False


class Table(BaseModel):
    rows: Sequence[Row] = Field(default_factory=list)
    num_rows: Optional[int]
    num_cols: Optional[int]


class Page(BaseModel):
    tables: Sequence[Table] = Field(default_factory=list)


class OcrTablesAsyncDataClass(BaseModel):
    pages: Sequence[Page] = Field(default_factory=list)
    num_pages: Optional[int]
