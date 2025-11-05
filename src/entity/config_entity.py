from datetime import datetime
import os
import sys

from src.constant.training_pipeline import *

print(ARTIFACT_DIR)
print(PIPLINE_NAME)

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=PIPLINE_NAME
        self.artifact_name=ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.timestamp=timestamp


class DataIngestionConfig:
    def __init__(self,trainig_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir=os.path.join(trainig_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
        self.traing_file_path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)
        self.test_file_path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
        self.train_test_ratio:float=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.data_base_name=DATA_INGESTION_DATABASE_NAME
        self.collection_name=DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir:str=os.path.join(self.data_validation_dir,DATA_VALIDATION_VALID_DIR)
        self.invalid_dir:str=os.path.join(self.data_validation_dir,DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path:str=os.path.join(self.valid_data_dir,TRAIN_FILE_NAME)
        self.valid_test_file_path:str=os.path.join(self.valid_data_dir,TEST_FILE_NAME)
        self.invalid_train_file_path:str=os.path.join(self.invalid_dir,TRAIN_FILE_NAME)
        self.invalid_test_file_path:str=os.path.join(self.invalid_dir,TEST_FILE_NAME)
        self.drift_report_file_path:str=os.path.join(self.data_validation_dir,
                                               DATA_VALIADATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,)
        

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir:str =os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORAMATION_DIR)
        self.transformed_train_file_path:str =os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DIR,
                                                           TRAIN_FILE_NAME.replace("csv","npy"),)
        self.transformed_test_file_path:str =os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DIR,
                                                     TEST_FILE_NAME.replace("csv","npy"),)
        self.transformed_object_file_path:str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT,
                                                             PREPROCESSING_OBJECT_FILE_NAME,)
        

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.trained_model_dir:str =os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path:str = os.path.join(self.trained_model_dir,MODEL_TRAINER_TRAINED_MODEL,MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.expected_accuraccy:float  = MODEL_TRAINER_EXPECTED_SCORE
        self.over_fitting_under_fitting_threshold:float = MODEL_TRAINER_OVER_FIT_UNDER_FIT_THRESHOLD
