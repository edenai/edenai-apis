from io import BufferedReader
import json
from uuid import uuid4
import os
import datetime
from typing import Callable, Tuple
import boto3
from botocore.signers import CloudFrontSigner
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from settings import base_path, keys_path

BUCKET = ""
BUCKET_RESSOURCE = ""
CLOUDFRONT_KEY_ID = ""
REGION = "eu-west-3"
CLOUDFRONT_URL = "https://d14uq1pz7dzsdq.cloudfront.net/"

PROVIDER_PROCESS = "provider_process"
USER_PROCESS = "users_process"

URL_SHORT_PERIOD = 3600
URL_LONG_PERIOD  = 3600 * 24 * 7


def set_time_and_presigned_url_process(process_type: str) -> Tuple[Callable, int, str]:
    """ Returns A tuple with the adequat function to call, the url expiration time and the bucket to which
                the file will be uploaded, depending of the process type

    Args:
        process_type (str): Specifies the upload type, whether it's to generate an url for a provider, or for a user

    Returns:
        Tuple[Callable, int, str]: A tuple with the adequat function to call, the url expiration time and the bucket to which
                                    the file will be uploaded
    """
    if process_type == PROVIDER_PROCESS:
        return get_s3_file_url, URL_SHORT_PERIOD, BUCKET
    if process_type == USER_PROCESS:
        return get_cloud_front_file_url, URL_LONG_PERIOD, BUCKET_RESSOURCE
    

def rsa_signer(message):
    with open(os.path.join(keys_path, "cloudfront_private_key.pem"), 'rb') as key:
        private_key = serialization.load_pem_private_key(
            key.read(),
            password= None,
            backend= default_backend()
        )
    return private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())

def s3_client_load():
    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
    aws_access_key_id = api_settings["aws_access_key_id"]
    aws_secret_access_key = api_settings["aws_secret_access_key"]

    global BUCKET, BUCKET_RESSOURCE, CLOUDFRONT_KEY_ID
    BUCKET = api_settings["providers_resource_bucket"]
    BUCKET_RESSOURCE = api_settings["users_resource_bucket"]
    CLOUDFRONT_KEY_ID = api_settings["cloudfront_key_id"]
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def upload_file_to_s3(file_path: str, file_name: str, process_type = PROVIDER_PROCESS):
    """Upload file to s3"""
    filename = str(uuid4()) + "_" + str(file_name)
    s3_client = s3_client_load()
    func_call, process_time, bucket = set_time_and_presigned_url_process(process_type)
    s3_client.upload_file(file_path, bucket, filename)
    return func_call(filename, process_time)

def upload_file_bytes_to_s3(file : bytes, file_name: str, process_type = PROVIDER_PROCESS):
    """Upload file byte to s3"""
    filename = str(uuid4()) + "_" + str(file_name)
    s3_client = s3_client_load()
    func_call, process_time, bucket = set_time_and_presigned_url_process(process_type)
    s3_client.upload_fileobj(file, bucket, filename)
    return func_call(filename, process_time)


def get_cloud_front_file_url(filename: str, process_time: int):
    cloudfront_signer = CloudFrontSigner(CLOUDFRONT_KEY_ID, rsa_signer)

    signed_url = cloudfront_signer.generate_presigned_url(
        f"{CLOUDFRONT_URL}{filename}", 
        date_less_than= datetime.datetime.now() + datetime.timedelta(seconds= process_time)
    )
    return signed_url


def get_s3_file_url(filename: str, process_time: int):
    """Get url of a file hosted on s3"""
    s3_client = s3_client_load()
    response = s3_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET,
            "Key": filename,
        },
        ExpiresIn=process_time,
    )
    return response


def get_providers_json_from_s3():
    s3_client = s3_client_load()
    obj = s3_client.get_object(
        Bucket="providers-cost", Key="providers_cost_master.json"
    )
    json_dict = json.loads(obj["Body"].read().decode("utf-8"))
    return json_dict["cost_data"]
