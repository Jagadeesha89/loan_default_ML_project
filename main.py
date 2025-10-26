import sys
from src.exception.exception import LoanDefaultException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig


if __name__ == "__main__":
    try:
        trainig_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(trainig_pipeline_config=trainig_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info(f"Initiating Data_ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion sucessfully completed")
        print(data_ingestion_artifact)
        data_validatoin_config=DataValidationConfig(training_pipeline_config=trainig_pipeline_config)
        data_validataion=DataValidation(data_validataion_config=data_validatoin_config,data_ingestion_artifact=data_ingestion_artifact)
        logging.info(f"Initiating the data validation at main.py")
        data_validation_artifact=data_validataion.initiate_data_validataion()
        logging.info("Data validataion sucessfully completed")
        print(data_validation_artifact)
    except Exception as e:
        raise LoanDefaultException(e,sys)
    
        
