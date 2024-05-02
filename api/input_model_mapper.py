from typing import Optional, Any

import pandas as pd
from pydantic import create_model, BaseModel

from Draft import feature_builder_v2

# Assuming `df` is your pandas DataFrame
# For demonstration, let's create a sample DataFrame similar to your structure
data = {"term_parser__term": [1.0], "zip__zip_code": ["12345"]}
df = pd.DataFrame(data)
df["zip__zip_code"] = df["zip__zip_code"].astype("category")

# Map pandas dtypes to Pydantic types
dtype_mapping = {
    "float64": (Optional[float], None),
    "int64": (Optional[int], None),
    "object": (Optional[str], None),
    "category": (Optional[str], None),
}


# Function to generate Pydantic model from pandas DataFrame
def generate_pydantic_model_from_df(df, model_name="DynamicModel"):
    defaults = df.iloc[0]

    # Initialize the fields dictionary with types and defaults properly structured
    fields = {}
    for col in df.columns:
        dtype = df[col].dtype
        default_value = None if pd.isna(defaults[col]) else defaults[col]

        if dtype.name == 'category':
            field_type = (str, default_value)  # maintain as string but set default correctly
        elif dtype.name == 'int64':
            field_type = (int, default_value)
        elif dtype.name == 'float64':
            field_type = (float, default_value)
        else:
            field_type = (str, default_value)  # fallback to string for any unhandled types

        fields[col] = field_type

    # Define a custom dict method to handle potential data type conversion needs
    def custom_dict_method(self: BaseModel, **kwargs: Any) -> dict:
        original_dict = super(self.__class__, self).dict(**kwargs)
        return {key: original_dict[key] for key in original_dict}

    # Create the model with dynamically added fields
    model = create_model(model_name, **fields)
    model.dict = custom_dict_method  # Use the custom dict method

    return model


df = feature_builder_v2.load_datasets_and_prepare_features(
    drop_meta_data=True, ds_type=feature_builder_v2.DatasetType.BASE
)
# Not ideal and should be done automatically but we need to drop columns not used by this model
# TODO: store used column names in 'ml_config_core.ModelTrainingResult'

# Create the Pydantic model
DynamicModel = generate_pydantic_model_from_df(df, "LoanData")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float

    data: DynamicModel


class DatasetSampleRowPredictionResponse(BaseModel):
    selected_row: int
    actual_value: int

    data: PredictionResponse


if __name__ == "__main__":
    test_row = df.sample(n=1).to_dict(orient="records")[0]
    model_instance = DynamicModel(**test_row)
    print(model_instance.model_dump_json())
