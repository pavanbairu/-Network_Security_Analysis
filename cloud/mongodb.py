import boto3

def get_mongodb_url(parameter_name, region_name):
    # Create a session with AWS SSM
    ssm_client = boto3.client('ssm', region_name=region_name)
    
    # Fetch the parameter value
    response = ssm_client.get_parameter(
        Name=parameter_name,  # The path of the parameter
        WithDecryption=True   # Enable decryption for SecureString parameters
    )
    
    return response['Parameter']['Value']  # Return the parameter value