import yaml
from src.exception.exception import LoanDefaultException
from src.logging.logger import logging
import sys,os
import pandas as pd
import numpy as np
import pickle



def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb")as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise LoanDefaultException(e,sys)
    
def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise LoanDefaultException(e,sys) 
    
def save_numpy_array(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise LoanDefaultException(e,sys)

def save_object(file_path:str,obj:object)->None:
    try:
        logging.info(f"Enter the save object path")
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obje:
            pickle.dump(obj,file_obje)
        logging.info(f"Pickle file saved sucessfully")
    except Exception as e:
        raise LoanDefaultException(e,sys)
