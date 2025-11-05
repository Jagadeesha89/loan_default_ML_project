from src.entity.artifact_entity import ClassificationMetricsArtifact
from src.exception.exception import LoanDefaultException
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import sys,os

def get_classification_metrics(y_true,y_pred):
    try:
        model_f1_score = f1_score(y_true,y_pred)
        model_accuracy =accuracy_score(y_true,y_pred)
        model_recall_score =recall_score(y_true,y_pred)
        model_precision_score = precision_score(y_true,y_pred)

        classificationmetricsartifact = ClassificationMetricsArtifact(
            f1score=model_f1_score,
            precision_score=model_precision_score,
            accuracy_score=model_accuracy,
            recall_score=model_recall_score
        )

        return classificationmetricsartifact
    except Exception as e:
        raise LoanDefaultException(e,sys)