import sys
from src.exception.exception import LoanDefaultException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTranformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig


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
        data_transformation_config=DataTransformationConfig(training_pipeline_config=trainig_pipeline_config)
        data_transformation=DataTranformation(data_transformation_config=data_transformation_config,data_validation_artifact=data_validation_artifact)
        logging.info(f"Inititating the data transformation stage at main.py")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        model_trainer_config=ModelTrainerConfig(training_pipeline_config=trainig_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        logging.info(f"Initiating the model trainer stage at main.py")
        model_trainer_artifact=model_trainer.initiate_model_trainer()
    except Exception as e:
        raise LoanDefaultException(e,sys)
    
        
