import typing as t
import json
from flask import Flask, request
from werkzeug.exceptions import HTTPException
from loguru import logger
from model.predict import make_prediction




app = Flask(__name__)
@app.route("/predict", methods=['POST'])
def predict():
    input_df = request.get_json()

    results = make_prediction(input_data=input_df)

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results


def main():
    app.run(host='0.0.0.0')
