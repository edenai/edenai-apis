import asyncio
import datetime
import json
import os
from io import BytesIO
from typing import Callable, Tuple
from uuid import uuid4

import aioboto3
import aiofiles
import boto3
from botocore.signers import CloudFrontSigner
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.settings import keys_path

BUCKET = ""
BUCKET_RESSOURCE = ""
CLOUDFRONT_KEY_ID = ""
REGION = ""
CLOUDFRONT_URL = "https://d14uq1pz7dzsdq.cloudfront.net/"

PROVIDER_PROCESS = "provider_process"
USER_PROCESS = "users_process"

URL_SHORT_PERIOD = 3600
URL_LONG_PERIOD = 3600 * 24 * 7


def set_time_and_presigned_url_process(process_type: str) -> Tuple[Callable, int, str]:
    """Returns A tuple with the adequat function to call, the url expiration time and the bucket to which
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
    with open(os.path.join(keys_path, "cloudfront_private_key.pem"), "rb") as key:
        private_key = serialization.load_pem_private_key(
            key.read(), password=None, backend=default_backend()
        )
    return private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())


def s3_client_load():
    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
    aws_access_key_id = api_settings["aws_access_key_id"]
    aws_secret_access_key = api_settings["aws_secret_access_key"]

    global BUCKET, BUCKET_RESSOURCE, CLOUDFRONT_KEY_ID, REGION
    BUCKET = api_settings["providers_resource_bucket"]
    BUCKET_RESSOURCE = api_settings["users_resource_bucket"]
    CLOUDFRONT_KEY_ID = api_settings["cloudfront_key_id"]
    REGION = api_settings["ressource_region"]
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def upload_file_to_s3(file_path: str, file_name: str, process_type=PROVIDER_PROCESS):
    """Upload file to s3"""
    filename = str(uuid4()) + "_" + str(file_name)
    s3_client = s3_client_load()
    func_call, process_time, bucket = set_time_and_presigned_url_process(process_type)
    s3_client.upload_file(file_path, bucket, filename)
    return func_call(filename, process_time)


def upload_file_bytes_to_s3(
    file: BytesIO, file_name: str, process_type: str = PROVIDER_PROCESS
) -> str:
    """Upload file byte to s3"""
    filename = str(uuid4()) + "_" + str(file_name)
    s3_client = s3_client_load()
    func_call, process_time, bucket = set_time_and_presigned_url_process(process_type)
    s3_client.upload_fileobj(file, bucket, filename)
    return func_call(filename, process_time)


def get_cloud_front_file_url(filename: str, process_time: int) -> str:
    cloudfront_signer = CloudFrontSigner(CLOUDFRONT_KEY_ID, rsa_signer)

    signed_url = cloudfront_signer.generate_presigned_url(
        f"{CLOUDFRONT_URL}{filename}",
        date_less_than=datetime.datetime.now()
        + datetime.timedelta(seconds=process_time),
    )
    return signed_url


def get_s3_file_url(filename: str, process_time: int) -> str:
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


# Async S3 utilities
async def as3_client_load():
    """Create async S3 client using aioboto3"""
    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
    aws_access_key_id = api_settings["aws_access_key_id"]
    aws_secret_access_key = api_settings["aws_secret_access_key"]

    global BUCKET, BUCKET_RESSOURCE, CLOUDFRONT_KEY_ID, REGION
    BUCKET = api_settings["providers_resource_bucket"]
    BUCKET_RESSOURCE = api_settings["users_resource_bucket"]
    CLOUDFRONT_KEY_ID = api_settings["cloudfront_key_id"]
    REGION = api_settings["ressource_region"]

    session = aioboto3.Session()
    return session.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


async def aget_s3_file_url(filename: str, process_time: int) -> str:
    """Async version: Get url of a file hosted on s3"""
    async with await as3_client_load() as s3_client:
        response = await s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET,
                "Key": filename,
            },
            ExpiresIn=process_time,
        )
    return response


async def aget_cloud_front_file_url(filename: str, process_time: int) -> str:
    """Async version: Get CloudFront signed URL"""

    # Run the signing operation in a thread pool since it's CPU-bound
    def _sign():
        cloudfront_signer = CloudFrontSigner(CLOUDFRONT_KEY_ID, rsa_signer)
        signed_url = cloudfront_signer.generate_presigned_url(
            f"{CLOUDFRONT_URL}{filename}",
            date_less_than=datetime.datetime.now()
            + datetime.timedelta(seconds=process_time),
        )
        return signed_url

    return await asyncio.to_thread(_sign)


async def aupload_file_to_s3(
    file_path: str, file_name: str, process_type=PROVIDER_PROCESS
):
    """Upload file to s3"""
    filename = str(uuid4()) + "_" + str(file_name)
    if isinstance(file_path, str):
        async with aiofiles.open(file_path, "rb") as file_:
            file = await file_.read()
    else:
        file = file_path
    file = BytesIO(file)
    return await aupload_file_bytes_to_s3(file, filename, process_type)

    # s3_client = s3_client_load()
    # func_call, process_time, bucket = set_time_and_presigned_url_process(process_type)
    # s3_client.upload_file(file_path, bucket, filename)
    # return func_call(filename, process_time)


async def aupload_file_bytes_to_s3(
    file: BytesIO, file_name: str, process_type: str = PROVIDER_PROCESS
) -> str:
    """Async version: Upload file bytes to s3"""
    filename = str(uuid4()) + "_" + str(file_name)

    # Load API settings first to populate global variables
    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
    global BUCKET, BUCKET_RESSOURCE, CLOUDFRONT_KEY_ID, REGION
    BUCKET = api_settings["providers_resource_bucket"]
    BUCKET_RESSOURCE = api_settings["users_resource_bucket"]
    CLOUDFRONT_KEY_ID = api_settings["cloudfront_key_id"]
    REGION = api_settings["ressource_region"]

    # Now determine bucket and other settings
    func_call, process_time, bucket = set_time_and_presigned_url_process(process_type)

    # Choose the async version of the URL generation function
    if process_type == PROVIDER_PROCESS:
        async_func_call = aget_s3_file_url
    elif process_type == USER_PROCESS:
        async_func_call = aget_cloud_front_file_url
    else:
        async_func_call = aget_s3_file_url

    # Upload to S3 asynchronously
    async with await as3_client_load() as s3_client:
        await s3_client.upload_fileobj(file, bucket, filename)

    # Generate and return the URL
    return await async_func_call(filename, process_time)
