import typing as t
import json
import pandas as pd
from flask import Flask, request
from werkzeug.exceptions import HTTPException
from loguru import logger
from model.processing import validate_inputs
from model.predict import make_prediction




app = Flask(__name__)
@app.route("/predict", methods=['POST'])
def predict():

    input_df = pd.DataFrame(json.dumps(request.json))
    validated_data, errors = validate_inputs(input_data= input_df)

    results = make_prediction(input_data=validated_data)

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results


def main():
    app.run()
