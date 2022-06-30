from numpy import rad2deg
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from config import create_and_validate_config

config = create_and_validate_config()

beer_pipeline_RFR = Pipeline(steps=[

    ('scaler', MinMaxScaler()),

    ('RFRegressor/Hyperparameter tunning w/ randomizedsearchCV', RandomizedSearchCV(
        RandomForestRegressor(random_state=config.model_config.seed),
        param_distributions=config.model_config.model_param['RFR'],
        random_state=config.model_config.seed,
        n_jobs=-1,
        cv=5
    ))

    ])


beer_pipeline_SVR = Pipeline(steps=[
    ('scaler', MinMaxScaler()),

    ('SVRegressor/Hyperparameter tunning w/ randomizedsearchCV', RandomizedSearchCV(
    SVR(),
    param_distributions=config.model_config.model_param['SVR'],
    random_state=config.model_config.seed,
    n_jobs=-1,
    cv=5
))])

beer_pipeline_LR = Pipeline(steps=[
    ('scaler', MinMaxScaler()),

    ('lasso regression', Lasso())
    ])




