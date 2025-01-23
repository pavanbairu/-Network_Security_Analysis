import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from box import Box
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging
from networksecurity.entity.config_entity import DataValidationConfig, TrainingPipelineConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH, DATA_VALIDATION_THRESHOLD
from networksecurity.utils.common import read_yaml, write_yaml
from scipy.stats import ks_2samp

import logging

class DataValidation:

    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Initializes the DataValidation class.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Contains paths to train and test data files.
            data_validation_config (DataValidationConfig): Configuration object containing directories and file paths for validation.

        Returns:
            None
        """
        try:
            logging.info("Initializing DataValidation class.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config: Box = read_yaml(SCHEMA_FILE_PATH)
            logging.info(f"Schema configuration loaded from {SCHEMA_FILE_PATH}")

            # Create directories for storing valid and invalid data
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)
            logging.info("Validation and invalidation directories created successfully.")

        except Exception as e:
            logging.error("Error during DataValidation initialization.")
            raise NetworkSecurityException(e, sys)


    @staticmethod
    def read_dataset(file_path: Path) -> pd.DataFrame:
        """
        Reads a dataset from the given file path and returns it as a pandas DataFrame.

        Args:
            file_path (Path): Path to the CSV file.

        Returns:
            pd.DataFrame: The dataset read from the file.
        """
        try:
            logging.info(f"Reading dataset from file path: {file_path}")
            return pd.read_csv(file_path)
        
        except Exception as e:
            logging.error(f"Error while reading dataset from {file_path}")
            raise NetworkSecurityException(e, sys)
    

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates if the number of columns in the given DataFrame matches the expected number of columns from the schema.

        Args:
            dataframe (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the number of columns matches the expected schema, False otherwise.
        """
        try:
            required_columns = len(self._schema_config.columns)  # Expected number of columns
            df_columns = len(list(dataframe.columns))  # Number of columns in the DataFrame

            logging.info(f"Validating number of columns: Expected {required_columns}, Found {df_columns}")
            return required_columns == df_columns

        except Exception as e:
            logging.error("Error while validating the number of columns.")
            raise NetworkSecurityException(e, sys)
    

    def numerical_column_exists(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if the required numerical columns exist in the given DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all required numerical columns exist, False otherwise.
        """
        try:
            required_numeric_columns = len(self._schema_config.numerical_columns)  # Expected numeric columns
            df_numeric_columns = dataframe.select_dtypes(exclude=['object']).shape[1]  # Actual numeric columns

            logging.info(f"Validating numerical columns: Expected {required_numeric_columns}, Found {df_numeric_columns}")
            return required_numeric_columns == df_numeric_columns
            
        except Exception as e:
            logging.error("Error while validating numerical columns.")
            raise NetworkSecurityException(e, sys)

    def data_drift(self, base_df: pd.DataFrame,
                   current_df: pd.DataFrame,
                   threshold: float = DATA_VALIDATION_THRESHOLD) -> bool:
        """
        Checks for data drift between the base dataset and the current dataset using the Kolmogorov-Smirnov test.

        Args:
            base_df (pd.DataFrame): The reference dataset (usually training data).
            current_df (pd.DataFrame): The current dataset (usually test or new data).
            threshold (float): Threshold p-value for detecting drift. Default is set in configuration.

        Returns:
            bool: True if no significant drift is found, False otherwise.
        """
        try:
            logging.info("Starting data drift analysis.")
            status = True  # Default status: no drift
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_disn = ks_2samp(d1, d2)  # Perform KS test

                # Check if drift is detected
                is_found = threshold > is_same_disn.pvalue
                if is_found:
                    status = False  # Drift found for at least one column
                    logging.warning(f"Data drift detected for column: {column}. P-value: {is_same_disn.pvalue}")

                # Update drift report
                report.update({  
                    column: {
                        "pvalue": float(is_same_disn.pvalue),
                        "is_found": is_found
                    }}
                )

            # Write the drift report to YAML
            drift_report_path = self.data_validation_config.drift_report_file_path
            dir_name = os.path.dirname(drift_report_path)
            os.makedirs(dir_name, exist_ok=True)
            write_yaml(report, drift_report_path)
            logging.info(f"Data drift analysis completed. Drift report saved at {drift_report_path}")

            return status
        
        except Exception as e:
            logging.error("Error during data drift analysis.")
            raise NetworkSecurityException(e, sys)        


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Orchestrates the data validation process by validating column counts, numerical columns, and checking for data drift.

        Args:
            None

        Returns:
            DataValidationArtifact: Contains information about the validation results, including paths to valid/invalid datasets and drift reports.
        """
        try: 
            
            logging.info(f"{'>'*10} data validation {'<'*10}")
            
            # Load train and test datasets
            train_path = self.data_ingestion_artifact.trained_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            train_data = DataValidation.read_dataset(train_path)
            test_data = DataValidation.read_dataset(test_path)
            logging.info("Training and testing datasets successfully loaded.")

            # Validate column counts for train and test data
            if not self.validate_number_of_columns(train_data):
                logging.warning("The train dataset does not contain the required number of columns.")
            if not self.validate_number_of_columns(test_data):
                logging.warning("The test dataset does not contain the required number of columns.")

            # Validate numeric columns for train and test data
            if not self.numerical_column_exists(train_data):
                logging.warning("The training dataset does not contain the required numeric columns.")
            if not self.numerical_column_exists(test_data):
                logging.warning("The test dataset does not contain the required numeric columns.")

            # Check for data drift
            status = self.data_drift(base_df=train_data, current_df=test_data)

            # Handle valid and invalid datasets based on drift
            if not status:
                logging.warning("Data validation failed due to drift.")
                train_data.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_data.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                data_validation_artifact = DataValidationArtifact(
                    validation_status=status,
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )
            else:
                logging.info("Data validation passed successfully.")
                train_data.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_data.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                data_validation_artifact = DataValidationArtifact(
                    validation_status=status,
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )

            logging.info("Data validation process completed.")
            return data_validation_artifact
        
        except Exception as e:
            logging.error("Error during data validation process.")
            raise NetworkSecurityException(e, sys)
