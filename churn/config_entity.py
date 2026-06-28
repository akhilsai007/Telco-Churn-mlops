from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_path: Path
    ingested_data_path: Path


@dataclass
class DataValidationConfig:
    root_dir: Path
    report_path: Path
    status_path: Path
    target_column: str
    all_schema: dict          # column name -> expected dtype
    categorical_values: dict  # column name -> list of allowed values
    expected_rows: Optional[int] = None


@dataclass
class DataTransformationConfig:
    root_dir: Path
    preprocessor_path: Path
    train_path: Path
    test_path: Path
    test_size: float
    random_state: int
    target_column: str
    drop_columns: list


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    metrics_path: Path
    train_path: Path
    test_path: Path
    target_column: str
    experiment_name: str
    random_state: int