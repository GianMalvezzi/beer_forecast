import typing as t
import numpy as np
import pandas as pd
import glob
from model.config import create_and_validate_config, TRAINED_MODEL_DIR
from model.processing import load_pipeline
from model.processing import validate_inputs



config = create_and_validate_config()



rfr_file_name = glob.glob(str(TRAINED_MODEL_DIR / config.app_config.rfr_pipeline_save_file) + '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].*')
svr_file_name = glob.glob(str(TRAINED_MODEL_DIR / config.app_config.svr_pipeline_save_file) + '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].*')
lr_file_name = glob.glob(str(TRAINED_MODEL_DIR / config.app_config.lr_pipeline_save_file) + '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].*')


rfr_pipeline = load_pipeline(file_name=rfr_file_name[0])
svr_pipeline = load_pipeline(file_name=svr_file_name[0])
lr_pipeline = load_pipeline(file_name=lr_file_name[0])

def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": {'RFR': 0,'SVR': 0, 'LR': 0}, "errors": errors}

    if not errors:
        predictions_rfr = np.array(rfr_pipeline.predict(
            X=validated_data[config.model_config.features]))

        predictions_svr = np.array(rfr_pipeline.predict(
            X=validated_data[config.model_config.features]))

        predictions_lr = np.array(rfr_pipeline.predict(
            X=validated_data[config.model_config.features]))

        results = {
            "predictions": {'RFR': [pred for pred in predictions_rfr],'SVR': [pred for pred in predictions_svr], 'LR': [pred for pred in predictions_lr]},
            "errors": errors,
        }

    return results