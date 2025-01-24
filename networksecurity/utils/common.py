import os
import sys
import yaml
import numpy as np
import pickle
from pathlib import Path
from box import Box
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging

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
