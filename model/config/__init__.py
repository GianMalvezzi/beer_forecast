from pathlib import Path
from typing import Any, List
from pydantic import BaseModel
from envyaml import EnvYAML

import __main__

# Project Directories
PACKAGE_ROOT = Path(__file__).parent.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yaml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    beer_data: str
    holidays_data: str
    rfr_pipeline_save_file: str
    svr_pipeline_save_file: str
    lr_pipeline_save_file: str


class ModelConfig(BaseModel):
   model_param: dict
   seed: int
   features: List[str]
   target: str
   rfr_pipeline: str
   svr_pipeline: str
   lr_pipeline: str
   test_size: float
   beer_data: str
   holidays_data: str
   scoring: str
   variables_to_rename: dict


class Config(BaseModel):
    """Master config."""

    model_config: ModelConfig
    app_config: AppConfig

def find_config_file() -> Path:
    """Find the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> EnvYAML:
    """Parse YAML containing the package setup."""

    if cfg_path is None:
        cfg_path = find_config_file()

    if cfg_path:
        parsed_config = EnvYAML(cfg_path, strict= False)
        return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")



def create_and_validate_config(parsed_config: EnvYAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_config=ModelConfig(**parsed_config),
    )
    return _config
