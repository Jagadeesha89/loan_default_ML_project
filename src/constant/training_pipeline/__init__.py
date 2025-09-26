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




"""
Define the constant variable names for dataigestion

"""

DATA_INGESTION_DIR_NAME:str ='dataingestion'
DATA_INGESTION_DATABASE_NAME:str = "JAGA"
DATA_INGESTION_COLLECTION_NAME:str = "LoanDefaultdata"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_FEATURE_STORE_DIR:str = "features_store"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2