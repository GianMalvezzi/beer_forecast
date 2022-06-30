import typing as t
import numpy as np
import datetime
import joblib
import os
import pandas as pd
from datetime import date
from pathlib import Path
from sklearn.pipeline import Pipeline
from typing import List, Optional, Tuple
from pydantic import BaseModel, ValidationError
from ..config import DATASET_DIR, TRAINED_MODEL_DIR, create_and_validate_config

config = create_and_validate_config()


def load_datasets(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{config.model_config.beer_data}"), decimal=',')
    dataframe['Data'] = dataframe['Data'].astype('datetime64[ns]')
    dataframe['Consumo de cerveja (litros)'] = dataframe['Consumo de cerveja (litros)'].astype('float')
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    return transformed


def insert_holidays(transformed, file_name: str)-> pd.DataFrame:
    """
    Insert holidays column in the training dataset

    """
    df_holidays = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    df_holidays['DATE'] = df_holidays['DATE'].astype('datetime64[ns]')
    datelist = df_holidays.iloc[:,0].tolist()
    transformed.loc[transformed['date'].isin(datelist), 'is_holiday'] = 1
    transformed.loc[~transformed['date'].isin(datelist), 'is_holiday'] = 0
    return transformed



def save_pipeline(pipeline: str, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    if pipeline == config.model_config.rfr_pipeline:  
        save_file_name = config.app_config.rfr_pipeline_save_file + str(date.today().strftime("%Y%m%d"))+ ".pkl"

    if pipeline == config.model_config.svr_pipeline:    
        save_file_name = config.app_config.svr_pipeline_save_file + str(date.today().strftime("%Y%m%d"))+ ".pkl"
   
    if pipeline == config.model_config.lr_pipeline:    
        save_file_name = config.app_config.lr_pipeline_save_file + str(date.today().strftime("%Y%m%d"))+ ".pkl"
    
    save_path = TRAINED_MODEL_DIR / save_file_name
        
    joblib.dump(pipeline_to_persist, save_path)
    return save_path


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep

    for model_file in TRAINED_MODEL_DIR.iterdir():
        delete = True
        for path in do_not_delete:
            if path == model_file:
                delete = False
                break
        if delete:
            os.remove(model_file)


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    validated_data = input_data[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class BeerInputSchema(BaseModel):
    date: Optional[datetime.date]
    avg_temp: Optional[float]
    min_temp: Optional[float]
    max_temp: Optional[float]
    rainfall: Optional[float]
    is_weekend: Optional[int]
    beer_consumption: Optional[float]
    is_holiday: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[BeerInputSchema]