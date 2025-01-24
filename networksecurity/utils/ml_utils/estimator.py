import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging


class NetworkModel:
    """
    A wrapper class for machine learning models that integrates preprocessing 
    and prediction functionalities.
    """

    def __init__(self, model, preprocessor):
        """
        Initialize the NetworkModel with a trained model and a preprocessor.

        Args:
            model: The trained machine learning model used for prediction.
            preprocessor: The preprocessing object used to transform input data.
        """
        try:
            logging.info("Initializing NetworkModel with provided model and preprocessor.")
            self.model = model
            self.preprocessor = preprocessor
            logging.info("NetworkModel initialized successfully.")
        except Exception as e:
            logging.error(f"Error during NetworkModel initialization. Error: {e}")
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        """
        Predict the output for the given input data after preprocessing.

        Args:
            x (array-like): Input data to be transformed and used for prediction.

        Returns:
            y_pred (array-like): Predicted output.
        """
        try:

            x_transformed = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transformed)

            return y_pred
        
        except Exception as e:
            logging.error(f"Error during prediction. Error: {e}")
            raise NetworkSecurityException(e, sys)