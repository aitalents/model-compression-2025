from typing import Dict, Any

from pydantic import BaseModel, validator


class RecognitionSchema(BaseModel):
    score: float
    label: str
    metrics: Dict[str, Any]


class ResponseSchema(BaseModel):
    cardiffnlp: RecognitionSchema
    ivanlau: RecognitionSchema
    svalabs: RecognitionSchema
    EIStakovskii: RecognitionSchema
    jy46604790: RecognitionSchema

