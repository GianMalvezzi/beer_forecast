import numpy as np
from config import create_and_validate_config
from pipeline import beer_pipeline_RFR, beer_pipeline_SVR, beer_pipeline_LR
from processing import insert_holidays, load_datasets, save_pipeline, remove_old_pipelines
from sklearn.model_selection import train_test_split

config = create_and_validate_config()

def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_datasets(file_name=config.app_config.beer_data)
    data = insert_holidays(data, file_name=config.app_config.holidays_data)


    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features], 
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.seed,
    )

    # fit model
    beer_pipeline_RFR.fit(X_train, y_train)
    beer_pipeline_SVR.fit(X_train, y_train)
    beer_pipeline_LR.fit(X_train, y_train)

    # persist trained model
    rfr_path = save_pipeline(pipeline=config.model_config.rfr_pipeline, pipeline_to_persist=beer_pipeline_RFR)
    svr_path = save_pipeline(pipeline=config.model_config.svr_pipeline, pipeline_to_persist=beer_pipeline_SVR)
    lr_path = save_pipeline(pipeline=config.model_config.lr_pipeline, pipeline_to_persist=beer_pipeline_LR)
    
    model_paths = [rfr_path, svr_path, lr_path]

    remove_old_pipelines(files_to_keep= model_paths)

if __name__ == "__main__":
    run_training()