from io import BufferedReader
import json
from uuid import uuid4

import boto3
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


api_settings = load_provider(ProviderDataEnum.KEY, "amazon")

aws_access_key_id = api_settings["aws_access_key_id"]
aws_secret_access_key = api_settings["aws_secret_access_key"]
REGION = "eu-west-3"

s3_client = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
BUCKET = "providers-upload"


def upload_file_to_s3(file: BufferedReader, file_name: str):
    """Upload file to s3"""
    filename = str(uuid4()) + "_" + str(file_name)
    s3_client.upload_fileobj(file, BUCKET, filename)
    return get_s3_file_url(filename)


def get_s3_file_url(filename: str):
    """Get url of a file hosted on s3"""
    response = s3_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET,
            "Key": filename,
        },
        ExpiresIn=3600,
    )
    return response


def get_providers_json_from_s3():
    obj = s3_client.get_object(
        Bucket="providers-cost", Key="providers_cost_master.json"
    )
    json_dict = json.loads(obj["Body"].read().decode("utf-8"))
    return json_dict["cost_data"]
