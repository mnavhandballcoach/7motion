import os
import boto3
import json
from api.b2_client import hard_delete_file

S3_ENDPOINT = "https://s3.eu-central-003.backblazeb2.com"
BUCKET_NAME = os.getenv("VIDEOS_BUCKET", "handball-videos")

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET
)

def list_incoming():
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="incoming/")
    files = resp.get("Contents", []) or []
    video_exts = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in files if f["Key"].lower().endswith(video_exts)]
    return files, videos

def download_video_and_json(key):
    local_video = "input.mp4"
    local_json = "input.json"

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    with open(local_video, "wb") as f:
        f.write(obj["Body"].read())

    json_key = key.replace(".mp4", ".json")
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=json_key)
        with open(local_json, "wb") as f:
            f.write(obj["Body"].read())
    except Exception:
        with open(local_json, "w") as f:
            json.dump({"email": None, "name": None, "start": 0, "duration": 0}, f)

    return local_video, local_json, json_key

def upload_file(local_path, remote_name):
    s3.upload_file(local_path, BUCKET_NAME, remote_name)

def delete_incoming(key, json_key):
    hard_delete_file(key)
    hard_delete_file(json_key)
