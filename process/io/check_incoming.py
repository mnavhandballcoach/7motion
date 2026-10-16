"""
Verificação leve: há vídeos em incoming/?

Usa só boto3 (sem torch/ultralytics), por isso corre em segundos.
Escreve has_videos=true|false no GITHUB_OUTPUT para o workflow decidir
se vale a pena instalar tudo e processar.
"""
import os

import boto3

S3_ENDPOINT = "https://s3.eu-central-003.backblazeb2.com"
BUCKET_NAME = os.getenv("VIDEOS_BUCKET", "handball-videos")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def main():
    has_videos = "true"   # em caso de dúvida, deixa o processamento tentar

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="incoming/")
        files = resp.get("Contents", []) or []
        videos = [f for f in files if f["Key"].lower().endswith(VIDEO_EXTS)]

        print(f"incoming/: {len(videos)} vídeo(s)")
        has_videos = "true" if videos else "false"

    except Exception as e:
        print(f"Não foi possível verificar incoming/ ({e!r}); segue para o processamento.")

    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"has_videos={has_videos}\n")

    print(f"has_videos={has_videos}")


if __name__ == "__main__":
    main()
