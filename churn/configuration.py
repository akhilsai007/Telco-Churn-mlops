from pathlib import Path

from churn.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
)
from churn.utils import create_directories, read_yaml

CONFIG_FILE_PATH = Path("config/config.yaml")


class ConfigurationManager:
    """Reads config.yaml and builds typed config objects for each component."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config_path)
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        create_directories([cfg["root_dir"]])
        return DataIngestionConfig(
            root_dir=Path(cfg["root_dir"]),
            source_path=Path(cfg["source_path"]),
            ingested_data_path=Path(cfg["ingested_data_path"]),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self.config["data_validation"]
        schema = self.config["schema"]
        create_directories([cfg["root_dir"]])
        return DataValidationConfig(
            root_dir=Path(cfg["root_dir"]),
            report_path=Path(cfg["report_path"]),
            status_path=Path(cfg["status_path"]),
            target_column=schema["target_column"],
            all_schema=schema["columns"],
            categorical_values=schema["categorical_values"],
            expected_rows=schema.get("expected_rows"),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config["data_transformation"]
        schema = self.config["schema"]
        create_directories([cfg["root_dir"]])
        return DataTransformationConfig(
            root_dir=Path(cfg["root_dir"]),
            preprocessor_path=Path(cfg["preprocessor_path"]),
            train_path=Path(cfg["train_path"]),
            test_path=Path(cfg["test_path"]),
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
            target_column=schema["target_column"],
            drop_columns=cfg.get("drop_columns", []),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        dt = self.config["data_transformation"]
        schema = self.config["schema"]
        create_directories([cfg["root_dir"]])
        return ModelTrainerConfig(
            root_dir=Path(cfg["root_dir"]),
            model_path=Path(cfg["model_path"]),
            metrics_path=Path(cfg["metrics_path"]),
            train_path=Path(dt["train_path"]),
            test_path=Path(dt["test_path"]),
            target_column=schema["target_column"],
            experiment_name=cfg["experiment_name"],
            random_state=cfg["random_state"],
        )