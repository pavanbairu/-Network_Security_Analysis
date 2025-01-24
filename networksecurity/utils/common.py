import os
import sys
import yaml
import numpy as np
import pickle
from pathlib import Path
from box import Box
from typing import Any
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, r2_score, accuracy_score

def read_yaml(path: Path) -> dict:
    """
    Reads a YAML file from the specified path and returns its contents as a dictionary.

    Args:
        path (Path): The file path of the YAML file to read.

    Returns:
        dict: A dictionary representation of the YAML file contents.

    """
    try:

        with open(path, 'r') as file:
            data = yaml.safe_load(file)  # Safely load YAML content
            return Box(data)  # Convert the dictionary to a Box object for dot notation access
        
    except Exception as e:
        logging.error(f"Failed to read YAML file from path: {path}. Error: {e}")
        raise NetworkSecurityException(e, sys)


def write_yaml(data: dict, path: Path) -> dict:
    """
    Writes a dictionary to a YAML file at the specified path.

    Args:
        data (dict): The data to write into the YAML file.
        path (Path): The file path where the YAML file will be saved.

    Returns:
        dict: The data written to the YAML file.

    """
    try:

        with open(path, 'w') as file:
            yaml.dump(data, file)  # Write the dictionary to the file in YAML format
            return data
        
    except Exception as e:
        logging.error(f"Failed to write data to YAML file at path: {path}. Error: {e}")
        raise NetworkSecurityException(e, sys)


def save_numpy_array(data: np.array, path: Path):
    """
    Saves a NumPy array to a specified file path.

    Args:
        data (np.array): The NumPy array to be saved.
        path (Path): The file path where the array should be saved.

    Returns:
        None: The function does not return anything; it writes the array to the file.
    """
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)

        # Save NumPy array to file
        with open(path, 'wb') as file:
            np.save(path, data)
        
    except Exception as e:
        logging.error(f"Failed to save NumPy array at {path}. Error: {e}")
        raise NetworkSecurityException(e, sys)


def save_object(model, path: Path):
    """
    Saves a Python object (e.g., a machine learning model) to a specified file path using Pickle.

    Args:
        model (object): The object to be saved.
        path (Path): The file path where the object should be saved.

    Returns:
        None: The function does not return anything; it writes the object to the file.
    """
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)

        # Save the object to file
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    except Exception as e:
        logging.error(f"Failed to save object at {path}. Error: {e}")
        raise NetworkSecurityException(e, sys)
    

def load_numpy_array(path: Path):
    """
    loads a NumPy array to a specified file path.

    Args:   
        path (Path): The file path where the array should be saved.

    Returns:
        None: The function does not return anything; it loads writes the array to the file.
    """
    try:

        # load NumPy array to file
        with open(path, 'rb') as file:
            return np.load(file)
        
    except Exception as e:
        logging.error(f"Failed to save NumPy array at {path}. Error: {e}")
        raise NetworkSecurityException(e, sys)
    

def model_evaluation(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, models: dict, hyperparameters: dict) -> (dict, dict):
    """
    Evaluate multiple models using GridSearchCV to find the best hyperparameters, 
    calculate R2 scores for training and testing, and return performance metrics and fitted models.

    Args:
        X_train (np.array): Training input features.
        X_test (np.array): Testing input features.
        y_train (np.array): Training target labels.
        y_test (np.array): Testing target labels.
        models (dict): Dictionary of model names as keys and model instances as values.
        hyperparameters (dict): Dictionary of hyperparameter grids for each model.

    Returns:
        tuple: 
            - dict: A report containing model names as keys and test R2 scores as values.
            - dict: A dictionary containing model names as keys and their trained instances as values.
    """
    try:
        logging.info("Starting model evaluation with GridSearchCV for hyperparameter tuning.")

        report = {}  # To store model names and their respective test R2 scores.
        model_pickles = {}  # To store model names and their trained instances.

        for i, model_name in enumerate(models.keys()):

            model = models[model_name]
            param = hyperparameters[model_name]

            # Perform GridSearchCV to find the best parameters
            gs: GridSearchCV = GridSearchCV(model, param_grid=param, cv=3)
            gs.fit(X_train, y_train)
            best_params = gs.best_params_

            # Set best parameters and train the model
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            # Predict on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)

            # Update report and model_pickles
            report[model_name] = test_r2_score
            model_pickles[model_name] = model

        return (report, model_pickles)

    except Exception as e:
        logging.error(f"Error during model evaluation. Error: {e}")
        raise NetworkSecurityException(e, sys)
