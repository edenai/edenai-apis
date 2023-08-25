from pydantic import BaseModel, Strict


class TrainingModelMetrics(BaseModel):
    au_prc: float
    log_loss: float
    recall: float
    precision: float


class AutomlClassificationTraining(BaseModel):
    model_id: Strict
    metrics: TrainingModelMetrics = None
