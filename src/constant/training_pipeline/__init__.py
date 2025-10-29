import os
import sys
import numpy as np
import pandas as pd

"""
Defineing the common variables name for training pipeline

"""

TARGET_COLUMN="Default"
PIPLINE_NAME:str="LoanDefault"
ARTIFACT_DIR:str="Artifacts"
FILE_NAME:str="Loan_default.csv"

TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

SCHEMA_FILE_PATH:str = os.path.join('data_schema','schema.yaml')


"""
Define the constant variable names for dataigestion

"""

DATA_INGESTION_DIR_NAME:str ='dataingestion'
DATA_INGESTION_DATABASE_NAME:str = "JAGA"
DATA_INGESTION_COLLECTION_NAME:str = "LoanDefaultdata"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_FEATURE_STORE_DIR:str = "features_store"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2


"""
Define the constant variable names for datavalidation

"""

DATA_VALIDATION_DIR_NAME:str = "datavalidation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIADATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"

"""
Define the constant varibale for the datatransformation

"""

DATA_TRANSFORAMATION_DIR:str = "data_transfroamtion"
DATA_TRANSFORMATION_TRANSFORMED_DIR:str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT:str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME:str = "preprocesser.pkl"