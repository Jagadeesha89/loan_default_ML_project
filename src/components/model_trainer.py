import os,sys
import pandas as pd

from src.exception.exception import LoanDefaultException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.main_utils.utlis import save_object,load_object,evaluate_models,load_numpy_array
from src.utils.ml_utils.metrics.classification_metrics import get_classification_metrics
from src.utils.ml_utils.model.estimator import LoanDefaultModel

class ModelTrainer:
    def  __init__(self,model_trainer_config:ModelTrainerConfig,
                  data_transformation_artifact:DataTransformationArtifact):
        self.model_trainer_config= model_trainer_config
        self.data_transformation_artifact=data_transformation_artifact

    def train_model(self,x_train,y_train,x_test,y_test):
        models={
            "RandomForest_Classifier":RandomForestClassifier(verbose=1),
            "XGB_Classifier":XGBClassifier()
        }

        parmas={
            "RandomForest_Classifier":{
                #'min_samples_split':[2,3,4,5],
                #'min_samples_leaf':[1,2,4,8],
                #'n_estimators': [8,16,32,128,256]
                },

            "XGB_Classifier":{
                #'learning_rate':[.1,.01,.001],
                #'max_depth':[3,4,5,6],
                #'n_estimators':[50,100,200,300]
                }
        }

        model_report:dict=evaluate_models(x_train,y_train,x_test,y_test,models,parmas)
        best_model_score=max(sorted(model_report.values()))

        best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        logging.info(f"Model trainned sucessfully and the best model is {best_model_name} with accuracy {best_model_score}")

        best_model=models[best_model_name]

        y_train_predit=best_model.predict(x_train)
        classification_train_metric = get_classification_metrics(y_true=y_train,y_pred=y_train_predit)

        y_test_predict=best_model.predict(x_test)
        classification_test_metric = get_classification_metrics(y_true=y_test,y_pred=y_test_predict)

        preprocesser = load_object(filepath=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        loandefault_model=LoanDefaultModel(preprocesser=preprocesser,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=loandefault_model)


        model_trainer_artifact=ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            trained_metrics=classification_train_metric,
            test_metrics=classification_test_metric
        )

        return model_trainer_artifact
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Initiating the model training process")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_array =load_numpy_array(filepath=transformed_train_file_path)
            test_array = load_numpy_array(filepath=transformed_test_file_path)

            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_train_artifact =self.train_model(x_train=x_train,y_train=y_train,
                                                   x_test=x_test,y_test=y_test)
            
            logging.info(f"Model training process compeleted")

            return model_train_artifact
        except Exception as e:
            raise LoanDefaultException(e,sys)

