import json
from pathlib import Path

import yaml

from churn.logger import logger


def read_yaml(path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    logger.info(f"Loaded YAML config: {path}")
    return content


def create_directories(paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON report: {path}")
