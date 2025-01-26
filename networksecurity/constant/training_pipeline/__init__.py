import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "NetworkData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"
SAVED_MODEL_DIR =os.path.join("saved_models")

FINAL_PREPROCESSOR_PATH = "final-models/preprocessor.pkl"
FINAL_MODEL_PATH = "final-models/model.pkl"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "ml-cluster"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
DATA_INGESTION_RANDOM_STATE: int = 42


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_THRESHOLD: float = 0.05

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

MODELS: dict = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighborsClassifier" : KNeighborsClassifier(),
    "DecisionTreeClassifier" : DecisionTreeClassifier(),
    "RandomForestClassifier" : RandomForestClassifier(),
    "GradientBoostingClassifier" : GradientBoostingClassifier(),
    "AdaBoostClassifier" : AdaBoostClassifier()
}

HYPERPARAMETERS: dict = {
    "LogisticRegression": {},
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy", "log_loss"],
#         "splitter": ["best", "random"],
#         "max_depth": [None, 5, 10, 20, 50],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
    },
    "RandomForestClassifier": {
        "n_estimators": [10, 50, 100, 200],
#         "criterion": ["gini", "entropy", "log_loss"],
#         'max_features':['sqrt','log2',None],
#         "max_depth": [None, 10, 20, 50],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
#         "bootstrap": [True, False],
    },
    "GradientBoostingClassifier": {
#         "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2, 0.5],
        'subsample':[0.6,0.7,0.75,0.85,0.9],
        # 'criterion':['squared_error', 'friedman_mse'],
        # 'max_features':['auto','sqrt','log2'],
#         "max_depth": [3, 5, 10],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoostClassifier": {
        "n_estimators": [10, 50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
#         "algorithm": ["SAMME", "SAMME.R"],
    }
}


"""
Model Evalaution ralated constant start with MODE TRAINER VAR NAME
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME= "report.yaml"


MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR

TRAINING_BUCKET_NAME = "mynetworksecurity"
PREDICTION_BUCKET_NAME = "my-network-datasource"
PREDICTION_DIR="prediction"