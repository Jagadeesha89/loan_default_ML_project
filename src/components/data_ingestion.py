import os
import sys
import pandas as pd
import numpy as np
import pymongo
from src.logging.logger import logging
from src.exception.exception import LoanDefaultException
from src.constant import training_pipeline
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGODB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config=data_ingestion_config
        

    def export_collection_to_dataframe(self):
        try:
            logging.info(f"Exporting collection from mongodb {self.data_ingestion_config.collection_name} as dataframe")

            database_name=self.data_ingestion_config.data_base_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongodb_clinet=pymongo.MongoClient(MONGODB_URL)
            collection=self.mongodb_clinet[database_name][collection_name]
            
            logging.info(f"Collection sucessfully extracted and converting to dataframe")

            df=pd.DataFrame(list(collection.find()))
            if "_id_" in df.columns.to_list():
                df.drop(columns="_id_",inplace=True)
                df=df[:25000]
            logging.info(f"Collection sucessfully converted to dataframe with rows: {df.shape[0]} and columns: {df.shape[1]}")
            return df
        except Exception as e:
            raise LoanDefaultException(e,sys)

    def export_data_to_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path =os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            logging.info(f"features are extracted and stored sucessfully at {feature_store_file_path}")
            return dataframe
        except Exception as e:
            raise LoanDefaultException(e,sys)
        
    def train_test_split_file(self,dataframe:pd.DataFrame):
        try:
            logging.info(f"entered train test split")
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_ratio)

            dir_path=os.path.dirname(self.data_ingestion_config.traing_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting test file path and train file")
            train_set.to_csv(self.data_ingestion_config.traing_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)

            logging.info(f"train test file exported sucessfully")

        except Exception as e:
            raise LoanDefaultException(e,sys)

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info(f"Data ingestion started")
            dataframe=self.export_collection_to_dataframe()
            dataframe=self.export_data_to_feature_store(dataframe=dataframe)
            self.train_test_split_file(dataframe=dataframe)
            data_ingestion_artifact=DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.traing_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data ingestion artifact sucessfully created")
            return data_ingestion_artifact
        except Exception as e:
            raise LoanDefaultException(e,sys)
        







