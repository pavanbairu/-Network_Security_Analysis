import os
import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.logger.logger import logging  # Assuming a logging setup is in place

if __name__ == "__main__":
    try:

        # Step 1: Initialize Training Pipeline Configuration
        training_pipeline_config = TrainingPipelineConfig()
        
        # Step 2: Initialize Data Ingestion Configuration
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        
        # Step 3: Initialize Data Ingestion Component
        data_ingestion = DataIngestion(data_ingestion_config)
        
        # Step 4: Execute Data Ingestion Process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Output the result of the data ingestion process
        print(data_ingestion_artifact)
        
    except Exception as e:
        # Log the exception and raise it for further debugging
        logging.error(f"An error occurred in the data ingestion pipeline: {str(e)}")
        raise e
