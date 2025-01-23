import os
import sys
import yaml
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
