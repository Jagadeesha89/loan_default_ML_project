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
