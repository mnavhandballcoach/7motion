import os
import boto3
from ultralytics import YOLO

S3_ENDPOINT = "https://s3.eu-central-003.backblazeb2.com"

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "tracking-backend-models")
MODEL_KEY = os.getenv("MODEL_KEY", "best.pt")
CLASSIFIER_KEY = os.getenv("CLASSIFIER_KEY", "bestclassifier.pt")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET
)

def download_from_b2(bucket, key, local_path):
    s3.download_file(bucket, key, local_path)
    return local_path

def load_yolo_model():
    local_path = download_from_b2(MODEL_BUCKET, MODEL_KEY, "best.pt")
    return YOLO(local_path)

def load_team_classifier():
    local_path = download_from_b2(MODEL_BUCKET, CLASSIFIER_KEY, "bestclassifier.pt")
    return YOLO(local_path)
