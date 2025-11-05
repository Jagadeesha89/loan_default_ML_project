import sys,os
from src.exception.exception import LoanDefaultException
from src.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME


class LoanDefaultModel:
    def __init__(self,preprocesser,model):
        try:
            self.preprocesser = preprocesser
            self.model=model
        except Exception as e:
            raise LoanDefaultException(e,sys)
        
    def predict(self,X):
        try:
            x_transform = self.preprocesser.transform(X)
            y_pred = self.model.predict(x_transform)
            return y_pred
        except Exception as e:
            raise LoanDefaultException(e,sys)
    
        