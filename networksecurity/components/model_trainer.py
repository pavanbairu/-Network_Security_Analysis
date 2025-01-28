import os
import sys
import numpy as np
import pandas as pd
import mlflow

from pathlib import Path
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constant.training_pipeline import MODELS, HYPERPARAMETERS
from networksecurity.logger.logger import logging
from networksecurity.utils.common import save_object, load_numpy_array, model_evaluation
from networksecurity.utils.ml_utils.classification_scores import get_classification_scores

import dagshub
dagshub.init(repo_owner='pavanbairu', repo_name='Network_Security_Analysis', mlflow=True)

from urllib.parse import urlparse

MLFLOW_TRACKING_URI = "https://dagshub.com/pavanbairu/Network_Security_Analysis.mlflow"
MLFLOW_TRACKING_USERNAME = "pavanbairu"
MLFLOW_TRACKING_PASSWORD = "45684133603ce329e9ccd4a9897b4e9d6f2176b0"

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

class ModelTrainer:

    def __init__(self,
                 data_transformer_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer class with data transformation artifacts and model trainer configuration.

        Args:
            data_transformer_artifact (DataTransformationArtifact): Artifact containing transformed train and test data paths.
            model_trainer_config (ModelTrainerConfig): Configuration for model training, including paths and parameters.
        """
        try:
            self.data_transformer_artifact = data_transformer_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            logging.error(f"Error initializing ModelTrainer. Error: {e}")
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self,best_model,classificationmetric):

        try: 
            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                f1_score=classificationmetric.f1_score
                precision_score=classificationmetric.precision_score
                recall_score=classificationmetric.recall_score
                
                mlflow.log_metric("f1_score",f1_score)
                mlflow.log_metric("precision",precision_score)
                mlflow.log_metric("recall_score",recall_score)
                mlflow.sklearn.log_model(best_model,"model")

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            logging.error(f"Error initializing ModelTrainer. Error: {e}")
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        """
        Train models, evaluate their performance, and save the best-performing model.

        Args:
            X_train (np.array): Training input features.
            y_train (np.array): Training target labels.
            X_test (np.array): Testing input features.
            y_test (np.array): Testing target labels.

        Returns:
            ModelTrainerArtifact: Artifact containing the best model path and classification metrics.
        """
        try:

            models: dict = MODELS
            hyperparameters: dict = HYPERPARAMETERS

            # Evaluate models with hyperparameters
            models_report, model_pickles = model_evaluation(X_train, X_test, y_train, y_test, models, hyperparameters)
            logging.info(f"Model evaluation completed. Models' performance report: {models_report}")

            # Select the best model
            best_model_score = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[list(models_report.values()).index(best_model_score)]
            best_model_pickle = model_pickles[best_model_name]
            logging.info(f"Best model identified: {best_model_name} with score {best_model_score}")

            # Calculate classification metrics
            y_train_pred = best_model_pickle.predict(X_train)
            train_classification_metric = get_classification_scores(y_train, y_train_pred)

            y_test_pred = best_model_pickle.predict(X_test)
            test_classification_metric = get_classification_scores(y_test, y_test_pred)
            logging.info("Classification metrics artifact created successfully.")

            self.track_flow(best_model_pickle, train_classification_metric)
            self.track_flow(best_model_pickle, test_classification_metric)

            # Save the best model
            dirname = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(dirname, exist_ok=True)
            save_object(best_model_pickle, self.model_trainer_config.trained_model_file_path)
            save_object(best_model_pickle, self.model_trainer_config.final_trained_model_path)
            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Create and return the artifact
            model_trainer_artifact = ModelTrainerArtifact(
                self.model_trainer_config.trained_model_file_path,
                train_classification_metric,
                test_classification_metric
            )
            
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error during model training. Error: {e}")
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Load transformed data, split into features and labels, and start the model training process.

        Returns:
            ModelTrainerArtifact: Artifact containing the best model path and classification metrics.
        """
        try:
            logging.info(f"{'>'*10} Model Training Process {'<'*10}")
            # Load train and test data
            train_data = load_numpy_array(self.data_transformer_artifact.transformed_train_file_path)
            test_data = load_numpy_array(self.data_transformer_artifact.transformed_test_file_path)

            # Split data into features and labels
            X_train, y_train, X_test, y_test = (
                train_data[:, :-1],
                train_data[:, -1],
                test_data[:, :-1],
                test_data[:, -1]
            )
            logging.info("Data loaded and split into features and labels.")

            # Train models and create an artifact
            model_trainer_artifact = self.train_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            logging.info(f"Model trainer process completed successfully.  Model Trainer Artifact = {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error during the model trainer process. Error: {e}")
            raise NetworkSecurityException(e, sys)
