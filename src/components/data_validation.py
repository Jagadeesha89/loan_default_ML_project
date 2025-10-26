import os
import sys
import pandas as pd
from src.exception.exception import LoanDefaultException
from src.logging.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils.utlis import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self,data_validataion_config:DataValidationConfig
                 ,data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"Initializing the data validation class")
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validataion_config=data_validataion_config
            self.schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise LoanDefaultException(e,sys)
        
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            logging.info(f"Reading the data from the file_path: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise LoanDefaultException(e,sys)
        
    def validate_data(self,dataframe:pd.DataFrame)->bool:
        try:
            logging.info(f"Enter the data validataion stage")
            number_of_columns=len([list(item.keys())[0] for item in self.schema_config['columns']])
            logging.info(f"Required Number of columns: {number_of_columns}")
            logging.info(f"Number of columns in the dataframe: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return  True
            return False
        except Exception as e:
            raise LoanDefaultException(e,sys)
        
    def detect_data_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report = {}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_sample_dict=ks_2samp(d1,d2)
                if threshold < is_sample_dict.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({
                    column:{
                        "p_value":float(is_sample_dict.pvalue),
                        "drift_status": is_found
                    }
                })

            drift_report_file_path=self.data_validataion_config.drift_report_file_path

            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(drift_report_file_path,content=report)
            logging.info(f"Data drift report is saved at {drift_report_file_path}")
        except Exception as e:
            raise LoanDefaultException(e,sys)
        
    def initiate_data_validataion(self)->DataValidationArtifact:
        try:
            logging.info(f"Initiating the data Validation process")
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path   

            train_df=DataValidation.read_data(train_file_path)
            test_df=DataValidation.read_data(test_file_path)

            status=self.validate_data(train_df)
            if not status:
                error_message= f"Train Dataframe does not contain all the required columns,\n"

            status=self.validate_data(test_df)
            if not status:
                error_message = f"Test Dataframe does not contain all the required  columns,\n"

            status=self.detect_data_drift(base_df=train_df,current_df=test_df)
            dir_path=os.path.dirname(self.data_validataion_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_df.to_csv(self.data_validataion_config.valid_train_file_path,index=False)
            test_df.to_csv(self.data_validataion_config.valid_test_file_path,index=False)

            logging.info(f"valid train file and test files are saved")

            data_validation_artifact=DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validataion_config.valid_train_file_path,
                valid_test_file_path=self.data_validataion_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validataion_config.drift_report_file_path
            ) 
            logging.info(f"Data validation artifact successfully created")
            
            return data_validation_artifact
            
        
        except Exception as e:
            raise LoanDefaultException(e,sys)