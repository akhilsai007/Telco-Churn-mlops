from churn.components.data_ingestion import DataIngestion
from churn.components.data_transformation import DataTransformation
from churn.components.data_validation import DataValidation
from churn.components.model_trainer import ModelTrainer
from churn.configuration import ConfigurationManager
from churn.logger import logger


def run_stage_one():
    config = ConfigurationManager()

    ingestion = DataIngestion(config.get_data_ingestion_config())
    ingested_path = ingestion.run()

    validation = DataValidation(config.get_data_validation_config())
    validation.run(ingested_path)

    transformation = DataTransformation(config.get_data_transformation_config())
    transformation.run(ingested_path)

    trainer = ModelTrainer(config.get_model_trainer_config())
    trainer.run()

    logger.info("Stages 1-3 (ingestion + validation + transformation + training) completed.")


if __name__ == "__main__":
    try:
        run_stage_one()
    except Exception as e:
        logger.exception(f"Stage 1 failed: {e}")
        raise