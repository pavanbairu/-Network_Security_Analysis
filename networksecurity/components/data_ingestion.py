import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging

from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataIngestionConfig
from cloud.mongodb import get_mongodb_url
from networksecurity.constant.training_pipeline import MONGOD_URL_PATH, AWS_REGION

import numpy as np
import pandas as pd
import pymongo

from sklearn.model_selection import train_test_split


class DataIngestion:
    """
    A class to handle the ingestion of data from a MongoDB database, 
    exporting it to feature stores, and splitting it into train and test datasets.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion object with configuration parameters.
        """
        self.data_ingestion_config = data_ingestion_config
        self.mongo_db_url = get_mongodb_url(MONGOD_URL_PATH, AWS_REGION)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Exports a MongoDB collection as a pandas DataFrame.
        Drops the '_id' column if it exists and replaces 'na' with NaN.

        Returns:
            DataFrame containing the exported data.
        """
        try:
            logging.info("Connecting to MongoDB to export collection as a DataFrame.")
            collection_name = self.data_ingestion_config.collection
            database_name = self.data_ingestion_config.database_name
            self.mongo_client = pymongo.MongoClient(self.mongo_db_url)
            collection = self.mongo_client[database_name][collection_name]
            
            logging.info(f"Fetching data from database: {database_name}, collection: {collection_name}")
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                logging.info("Dropping '_id' column from DataFrame.")
                df = df.drop(columns=["_id"], axis=1)

            logging.info("Replacing 'na' values with NaN in DataFrame.")
            df.replace({"na": np.nan}, inplace=True)

            logging.info("Successfully exported collection as DataFrame.")
            return df

        except Exception as e:
            logging.error("Error occurred while exporting collection as DataFrame.")
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Exports the DataFrame into a feature store by saving it as a CSV file.

        Args:
            dataframe (pd.DataFrame): DataFrame to export.

        Returns:
            The same DataFrame after export.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Exported DataFrame to feature store: {feature_store_file_path}")

            return dataframe

        except Exception as e:
            logging.error("Error occurred while exporting data to feature store.")
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Splits the DataFrame into training and testing datasets and saves them to CSV files.

        Args:
            dataframe (pd.DataFrame): DataFrame to split.
        """
        try:

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=self.data_ingestion_config.random_state
            )

            logging.info("Train-test split completed successfully.")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            logging.info(f"Exported training data to: {self.data_ingestion_config.training_file_path}")
            
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info(f"Exported testing data to: {self.data_ingestion_config.test_file_path}")


        except Exception as e:
            logging.error("Error occurred during train-test split.")
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by:
        1. Exporting the collection as a DataFrame.
        2. Saving the data into a feature store.
        3. Splitting the data into training and testing datasets.

        Returns:
            DataIngestionArtifact containing file paths for train and test datasets.
        """
        try:
            logging.info(f'{">"*10} data ingestion pipeline {"<"*10}')
            dataframe = self.export_collection_as_dataframe()
            
            dataframe = self.export_data_into_feature_store(dataframe)

            self.split_data_as_train_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data ingestion process completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            logging.error("Error occurred during data ingestion process.")
            raise NetworkSecurityException(e, sys)
