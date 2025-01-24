import os
import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import (TrainingPipelineConfig,
                                                  DataIngestionConfig,
                                                  DataValidationConfig,
                                                  DataTransformationConfig,
                                                  ModelTrainerConfig)

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


        # Initialize DataValidationConfig with training pipeline settings.
        data_validation_config = DataValidationConfig(training_pipeline_config)

        # Create a DataValidation instance with ingestion artifacts and validation config.
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                        data_validation_config=data_validation_config)

        # Perform data validation (column checks, numerical checks, and data drift).
        data_validation_artifact = data_validation.initiate_data_validation()

        # Configure data transformation settings.
        data_transformation_config = DataTransformationConfig(training_pipeline_config)

        # Initialize the DataTransformation process.
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)

        # Start the data transformation workflow.
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        # Initialize the model trainer configuration with the training pipeline configuration
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)

        # Create an instance of the ModelTrainer class using data transformation artifacts and the model trainer configuration
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)

        # Start the model training process and generate the model trainer artifact
        model_trainer_artifact = model_trainer.initiate_model_trainer()

    except Exception as e:
        # Log the exception and raise it for further debugging
        logging.error(f"An error occurred in the data ingestion pipeline: {str(e)}")
        raise e
