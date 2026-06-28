import pandas as pd

from churn.config_entity import DataIngestionConfig
from churn.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self) -> str:
        logger.info("===== Stage: Data Ingestion =====")
        source = self.config.source_path
        if not source.exists():
            raise FileNotFoundError(
                f"Raw data not found at '{source}'. Download the Telco Customer "
                f"Churn CSV from Kaggle (dataset 'blastchar/telco-customer-churn') "
                f"and place it there."
            )

        df = pd.read_csv(source)
        logger.info(f"Read raw data: {df.shape[0]} rows x {df.shape[1]} columns")

        out = self.config.ingested_data_path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved ingested copy to: {out}")
        return str(out)
