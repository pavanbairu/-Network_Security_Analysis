import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.common import save_numpy_array, save_object
from networksecurity.logger.logger import logging
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        Initializes the DataTransformation class.

        Args:
            data_validation_artifact (DataValidationArtifact): Contains paths to validated datasets and validation status.
            data_transformation_config (DataTransformationConfig): Configuration settings for data transformation.
        """
        try:
            logging.info("Initializing DataTransformation.")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(path: Path) -> pd.DataFrame:
        """
        Reads a CSV file and loads it into a DataFrame.

        Args:
            path (Path): The file path to the CSV file.

        Returns:
            pd.DataFrame: Data loaded into a DataFrame.
        """
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.error(f"Error reading data from path: {path}")
            raise NetworkSecurityException(e, sys)

    def get_data_transformation_object(cls) -> Pipeline:
        """
        Creates a data transformation pipeline with the KNNImputer.

        Returns:
            Pipeline: A Scikit-learn pipeline object with the imputer step configured.
        """
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            return Pipeline(steps=[("imputer", imputer)])
        except Exception as e:
            logging.error("Error creating data transformation pipeline.")
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes the data transformation process, including:
        - Reading train and test datasets.
        - Splitting data into input features and target labels.
        - Applying the transformation pipeline.
        - Saving transformed data and the pipeline object.

        Returns:
            DataTransformationArtifact: Contains paths to transformed data files and the transformation object.
        """
        try:
            logging.info(f"{'>'*10} data transformation process. {'<'*10}")

            # Load datasets
            train_path = (
                self.data_validation_artifact.valid_train_file_path
                if self.data_validation_artifact.validation_status
                else self.data_validation_artifact.invalid_train_file_path
            )
            test_path = (
                self.data_validation_artifact.valid_test_file_path
                if self.data_validation_artifact.validation_status
                else self.data_validation_artifact.invalid_test_file_path
            )
            train_df = DataTransformation.read_data(train_path)
            test_df = DataTransformation.read_data(test_path)
            logging.info(f"Loaded train and test datasets from {train_path} and {test_path}.")

            # Process train and test data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Apply transformations
            processor = self.get_data_transformation_object()
            transformed_input_train_df = processor.fit_transform(input_feature_train_df)
            transformed_input_test_df = processor.transform(input_feature_test_df)
            logging.info("Data transformation pipeline applied successfully to train and test data.")

            # Combine features and save
            train_array = np.c_[transformed_input_train_df, np.array(target_feature_train_df)]
            test_array = np.c_[transformed_input_test_df, np.array(target_feature_test_df)]
            save_numpy_array(train_array, self.data_transformation_config.transformed_train_file_path)
            save_numpy_array(test_array, self.data_transformation_config.transformed_test_file_path)
            save_object(processor, self.data_transformation_config.transformed_object_file_path)
            save_object(processor, "final-models/preprocessor.pkl")

            logging.info("Transformed data and pipeline object saved successfully.")

            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info("Data transformation process completed successfully.")
            return data_transformation_artifact

        except Exception as e:
            logging.error("Error occurred during the data transformation process.")
            raise NetworkSecurityException(e, sys)
