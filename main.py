import sys
from src.exception.exception import LoanDefaultException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig


if __name__ == "__main__":
    try:
        trainig_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(trainig_pipeline_config=trainig_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info(f"Initiating Data_ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion sucessfully completed")
    except Exception as e:
        raise LoanDefaultException(e,sys)
    
        
