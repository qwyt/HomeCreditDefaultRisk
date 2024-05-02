import json
import random

from typing import Union, List, Optional, Any, Dict

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from starlette.responses import JSONResponse

from api.input_model_mapper import DynamicModel, PredictionResponse, df, DatasetSampleRowPredictionResponse
from api.model_builder import build_model
from shared import ml_config_core

app = FastAPI()

model: ml_config_core.ModelTrainingResult = None


@app.on_event("startup")
async def load_model():
    print("--Loading Model")
    global model
    model = build_model()
    print("--Model Loaded")


def model_to_dict(model_instance: BaseModel) -> Dict[str, Any]:
    """
    Recursively convert a Pydantic model instance (and nested instances) to a dictionary
    with all values cast to strings.
    """
    result = {}
    for field, value in model_instance.dict().items():
        if isinstance(value, BaseModel):
            result[field] = model_to_dict(value)
        elif isinstance(value, list):
            result[field] = [model_to_dict(item) if isinstance(item, BaseModel) else str(item) for item in value]
        else:
            result[field] = str(value) if value is not None else None
    return result


# @app.post("/predict_dataset_row")
# def predict_sample_row(row_number: Optional[int] = None):
#     """
#
#     :param row_number: select a row to predict from the load sample dataset, leave None to select random row
#     :return:
#     """
#
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model is not loaded yet")
#
#     if row_number is not None and len(df) < row_number:
#         raise HTTPException(status_code=503, detail=f"Selected row: {row_number}, but sample only has: {len(df)} rows")
#
#     if row_number is None:
#         row_number = random.randint(0, len(df) - 1)
#     row = df.iloc[row_number:row_number + 1]
#
#     row_cp = row.copy()
#     prediction = model.test_data.test_model.predict(row_cp)[0].item()
#     probability = round(model.test_data.test_model.predict_proba(row_cp)[0][1].item(), 2)
#
#     row_dict = row.to_dict(orient='records')[0]
#     actual_value = row_dict["TARGET"]
#     data = DynamicModel(**row_dict)
#
#     r = PredictionResponse(data=data,
#                            prediction=prediction,
#                            probability=probability)
#     resp = DatasetSampleRowPredictionResponse(data=r, selected_row=row_number, actual_value=actual_value)
#     return model_to_dict(resp)


@app.post("/predict", response_model=PredictionResponse)
def predict_row(data: DynamicModel):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    data_df = pd.DataFrame([data.dict()])
    prediction = model.test_data.test_model.predict(data_df)[0].item()
    probability = round(model.test_data.test_model.predict_proba(data_df)[0][1].item(), 2)

    return PredictionResponse(data=data, prediction=prediction, probability=probability)


# @app.post("/predict/multiple", response_model=List[PredictionResponse])
# def predict_multiple(data: List[DynamicModel]):
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model is not loaded yet")
#
#     data_df = pd.DataFrame([item.dict() for item in data])
#     predictions = model.test_data.test_model.predict(data_df)
#     return [
#         PredictionResponse(data=item, prediction=pred.item())
#         for item, pred in zip(data, predictions)
#     ]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
