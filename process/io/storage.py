import os
import boto3
import json
from datetime import datetime, timezone

from botocore.exceptions import ClientError
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

# Fila de vídeos:
#   incoming/    -> vídeos novos (enviados pelo site)
#   processing/  -> vídeo reservado por uma corrida em curso
#   failed/      -> vídeos que falharam (vídeo + json + logs/<nome>.log)
INCOMING_PREFIX = "incoming/"
PROCESSING_PREFIX = "processing/"
FAILED_PREFIX = "failed/"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def json_key_for(key):
    """
    Chave do json que acompanha o vídeo (mesmo nome, extensão .json).
    Antes só funcionava para .mp4: noutros formatos o json_key ficava igual
    ao do vídeo.
    """
    return os.path.splitext(key)[0] + ".json"


def _exists(key):
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _move(src, dst, optional=False):
    """Copia no servidor (sem descarregar) e só depois apaga a origem."""
    if optional and not _exists(src):
        return False

    s3.copy_object(
        Bucket=BUCKET_NAME,
        Key=dst,
        CopySource={"Bucket": BUCKET_NAME, "Key": src},
    )
    hard_delete_file(src)
    return True


def list_incoming():
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=INCOMING_PREFIX)
    files = resp.get("Contents", []) or []
    videos = [f for f in files if f["Key"].lower().endswith(VIDEO_EXTS)]
    return files, videos


def claim_incoming(key):
    """
    Reserva o vídeo (e o json, se existir): incoming/ -> processing/.
    Devolve (chave_do_video, chave_do_json) já em processing/.
    """
    json_src = json_key_for(key)

    claimed_video = PROCESSING_PREFIX + os.path.basename(key)
    claimed_json = PROCESSING_PREFIX + os.path.basename(json_src)

    _move(key, claimed_video)
    _move(json_src, claimed_json, optional=True)

    return claimed_video, claimed_json


def download_video_and_json(key):
    local_video = "input.mp4"
    local_json = "input.json"

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    with open(local_video, "wb") as f:
        f.write(obj["Body"].read())

    json_key = json_key_for(key)
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


def finish_processing(video_key, json_key):
    """Sucesso: apaga o vídeo (e o json) de processing/."""
    hard_delete_file(video_key)
    if _exists(json_key):
        hard_delete_file(json_key)


def move_to_failed(video_key, json_key):
    """Erro: processing/ -> failed/ (vídeo e json)."""
    _move(video_key, FAILED_PREFIX + os.path.basename(video_key), optional=True)
    _move(json_key, FAILED_PREFIX + os.path.basename(json_key), optional=True)


def recover_stale_processing(max_age_minutes=360):
    """
    Itens que ficaram em processing/ há mais de `max_age_minutes` pertencem a
    uma corrida que morreu (timeout, falta de memória, cancelamento).
    Vão para failed/ para ficarem à vista e não se perderem.
    Devolve a lista de chaves movidas.
    """
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PROCESSING_PREFIX)
    now = datetime.now(timezone.utc)
    moved = []

    for obj in resp.get("Contents", []) or []:
        key = obj["Key"]
        if key == PROCESSING_PREFIX:
            continue

        age_min = (now - obj["LastModified"]).total_seconds() / 60.0
        if age_min < max_age_minutes:
            continue

        _move(key, FAILED_PREFIX + os.path.basename(key))
        moved.append(key)

    return moved


def delete_incoming(key, json_key):
    """Mantida por compatibilidade (apaga de incoming/ sem passar por processing/)."""
    hard_delete_file(key)
    hard_delete_file(json_key)
