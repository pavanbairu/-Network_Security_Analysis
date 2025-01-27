import os, sys


from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.model_trainer import ModelTrainer
from cloud.s3_sync import S3Sync
from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME, ARTIFACT_DIR, FINAL_MODELS
# from networksecurity.components.model_evaluation import ModelEvaluation
# from networksecurity.components.model_pusher import ModelPusher

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)

class TrainingPipeline:

    def __init__(self):
        """
        Initializes the TrainingPipeline with configuration settings for the entire pipeline.
        """
        self.training_pipeline_config = TrainingPipelineConfig()
        self.sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Handles the data ingestion process by initializing and running the data ingestion component.

        Args:
            None
        
        Returns:
            DataIngestionArtifact: Contains details of the ingested data.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            return data_ingestion_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Validates the ingested data by initializing and running the data validation component.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing details of the ingested data.
        
        Returns:
            DataValidationArtifact: Contains details of the validation process.
        """
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Transforms the validated data into a format suitable for model training by running the data transformation component.

        Args:
            data_validation_artifact (DataValidationArtifact): Artifact containing details of validated data.
        
        Returns:
            DataTransformationArtifact: Contains details of the transformation process.
        """
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformer_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Trains a machine learning model using the transformed data by running the model trainer component.

        Args:
            data_transformer_artifact (DataTransformationArtifact): Artifact containing transformed data.
        
        Returns:
            ModelTrainerArtifact: Contains details of the training process and the trained model.
        """
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformer_artifact=data_transformer_artifact,
                                         model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_evaluation(self) -> None:
        """
        Placeholder for model evaluation logic to assess the trained model's performance.

        Args:
            None
        
        Returns:
            None
        """
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_pusher(self) -> None:
        """
        Placeholder for model pusher logic to deploy the trained model to production.

        Args:
            None
        
        Returns:
            None
        """
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
        
    def sync_artifact_dir_to_s3(self):
        """
        Syncs the artifact directory to the specified AWS S3 bucket.
        
        Constructs the S3 bucket URL using the training pipeline's timestamp and
        uploads the entire artifact directory, ensuring that all artifacts are saved in S3 for future reference.

        Args:
            None
        
        Returns:
            None
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{ARTIFACT_DIR}/{self.training_pipeline_config.timestamp}"
            self.sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_s3_bucket_url=aws_bucket_url)
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
        
    def sync_saved_model_dir_to_s3(self):
        """
        Syncs the saved model directory to the specified AWS S3 bucket.
        
        Constructs the S3 bucket URL using the training pipeline's timestamp and
        uploads the saved models directory, ensuring the final models are securely stored in S3.

        Args:
            None
        
        Returns:
            None
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{FINAL_MODELS}/{self.training_pipeline_config.timestamp}"
            self.sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_s3_bucket_url=aws_bucket_url)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Executes the entire training pipeline sequentially:
        1. Data Ingestion
        2. Data Validation
        3. Data Transformation
        4. Model Training

        After training, the following additional steps are performed:
        - Sync the artifact directory to an S3 bucket for artifact storage.
        - Sync the saved model directory to an S3 bucket for model storage.

        Args:
            None
        
        Returns:
            ModelTrainerArtifact: Final artifact from the model training step.
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformer_artifact=data_transformation_artifact)

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
