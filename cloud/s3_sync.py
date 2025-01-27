import os

class S3Sync:
    """
    A class to handle synchronization of local folders with an AWS S3 bucket.
    Provides methods to upload a folder to S3 and download a folder from S3.
    """

    def sync_folder_to_s3(self, folder: str, aws_s3_bucket_url: str) -> None:
        """
        Syncs a local folder to an AWS S3 bucket using the AWS CLI.

        Args:
            folder (str): The local folder path to be synced to the S3 bucket.
            aws_s3_bucket_url (str): The URL of the AWS S3 bucket where the folder will be uploaded.

        Returns:
            None
        """
        command = f"aws s3 sync {folder} {aws_s3_bucket_url}"
        os.system(command=command)

    def sync_folder_from_s3(self, folder: str, aws_s3_bucket_url: str) -> None:
        """
        Syncs an AWS S3 bucket folder to a local folder using the AWS CLI.

        Args:
            folder (str): The local folder path where the S3 bucket's content will be downloaded.
            aws_s3_bucket_url (str): The URL of the AWS S3 bucket whose contents will be downloaded.

        Returns:
            None
        """
        command = f"aws s3 sync {aws_s3_bucket_url} {folder}"
        os.system(command=command)
