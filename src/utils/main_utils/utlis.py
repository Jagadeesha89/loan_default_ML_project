import yaml
from src.exception.exception import LoanDefaultException
from src.logging.logger import logging
import sys,os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score



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
    
def load_object(filepath:str)->object:
    try:
        logging.info(f"loading the model from the source folder")
        if not os.path.exists(filepath):
            raise Exception(f"provided {filepath} does not exist")
        with open(filepath,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise LoanDefaultException(e,sys)
    
def load_numpy_array(filepath:str)->np.array:
    try:
        logging.info(f"Loading the np.array file")
        if not os.path.exists(filepath):
            raise Exception(f"Provided file path does not exists")
        with open(filepath,'rb') as arry_file:
            return np.load(arry_file)
    except Exception as e:
        raise LoanDefaultException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        logging.info(f"Process enters evaluation of models")
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            parm=params[list(models.keys())[i]]
            gs=GridSearchCV(model,parm,cv=3)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            y_train_predict=model.predict(x_train)
            y_test_predict=model.predict(x_test)

            train_model_score=accuracy_score(y_train,y_train_predict)
            test_model_score=accuracy_score(y_test,y_test_predict)

            report[list(models.keys())[i]]=train_model_score

        return report
    except Exception as e:
        raise LoanDefaultException(e,sys)



