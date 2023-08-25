from pydantic import BaseModel, Strict


class AutomlClassificationCreateDataset(BaseModel):
    dataset_id: Strict
