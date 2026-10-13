import os
import cv2
import json
import time
import math
import boto3
import smtplib
import traceback
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from process.core.clustering_optimized import (
    crop_jersey_only,
    extract_embedding_fast,
    hsv_feature,
    run_initial_clustering,
    identify_team_clusters,
    decide_team_A_B,
    assign_cluster,
    assign_team
)

from process.core.tracking import (
    init_tracker,
    update_tracks,
    update_ema,
)

# ============================
# Configuração geral
# ============================

VIDEO_BUCKET = os.getenv("B2_BUCKET_NAME")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

N_CALIB_FRAMES = 200
EMA_DECAY = 0.92
TRACK_VOTE_WINDOW = 100

# ============================
# Helpers
# ============================

def log(msg):
    print(msg, flush=True)

def send_email(to_email, subject, body):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        log("SMTP not configured, skipping email.")
        return

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        log(f"Email sent to {to_email}")
    except Exception as e:
        log(f"Error sending email: {e}")


def download_video_from_b2(video_key, local_path):
    log(f"Downloading video {video_key} from B2 to {local_path}")
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.eu-central-003.backblazeb2.com",
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APP_KEY,
    )
    s3.download_file(VIDEO_BUCKET, video_key, local_path)


def upload_json_to_s3(data, key):
    log(f"Uploading JSON to S3: {key}")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    s3.put_object(
        Bucket=VIDEO_BUCKET,
        Key=key,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
    )


# ============================
# Processamento de vídeo
# ============================

def process_video(video_key, user_email):
    start_time = time.time()
    log(f"Starting process for video: {video_key}")

    local_video = "/tmp/input_video.mp4"
    download_video_from_b2(video_key, local_video)

    cap = cv2.VideoCapture(local_video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"Video FPS: {fps}, total frames: {total_frames}")

    tracker = init_tracker()
    team_memory = {}
    track_votes = {}

    embeddings = []
    bboxes = []
    centroids = None
    labels = None
    team_A = None
    team_B = None

    results = []

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ============================
        # 1. Detecção + tracking
        # ============================

        detections = []  # aqui entra o output do teu detector (YOLO, etc.)
        # detections: lista de dicts com keys: x1, y1, x2, y2, cls, score

        tracks = update_tracks(tracker, detections)

        frame_players = []

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]
            cls = t["cls"]

            if cls != "player":
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            player_info = {
                "track_id": track_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "team_raw": None,
                "team_final": None,
            }

            # ============================
            # 2. Calibração inicial (primeiros N_CALIB_FRAMES)
            # ============================

            if frame_index <= N_CALIB_FRAMES:
                jersey = crop_jersey_only(crop)
                emb = extract_embedding_fast(jersey)
                embeddings.append(emb)
                bboxes.append((x1, y1, x2, y2))
                frame_players.append(player_info)
                continue

            # ============================
            # 3. Clustering inicial (no frame N_CALIB_FRAMES + 1)
            # ============================

            if frame_index == N_CALIB_FRAMES + 1 and centroids is None:
                log("Running initial clustering...")
                centroids, labels = run_initial_clustering(np.array(embeddings))
                team_clusters = identify_team_clusters(labels)
                team_A, team_B = decide_team_A_B(team_clusters, bboxes, labels)
                log(f"Team clusters: A={team_A}, B={team_B}")

            # ============================
            # 4. Atribuição de equipa por embedding
            # ============================

            jersey = crop_jersey_only(crop)
            emb = extract_embedding_fast(jersey)

            cluster_id = assign_cluster(emb, centroids)
            team_raw = assign_team(cluster_id, team_A, team_B)

            # Fallback HSV quando embedding está instável
            if team_raw is None:
                hsv = hsv_feature(jersey)
                cluster_id = assign_cluster(hsv, centroids)
                team_raw = assign_team(cluster_id, team_A, team_B)

            player_info["team_raw"] = team_raw

            # ============================
            # 5. EMA + votação por track_id
            # ============================

            if track_id not in team_memory:
                team_memory[track_id] = None
                track_votes[track_id] = []

            team_temporal = team_raw
            update_ema(team_memory, track_id, team_temporal, decay=EMA_DECAY)

            if team_memory[track_id] is not None:
                track_votes[track_id].append(team_memory[track_id])

            if len(track_votes[track_id]) > TRACK_VOTE_WINDOW:
                track_votes[track_id].pop(0)

            if track_votes[track_id]:
                vals, counts = np.unique(track_votes[track_id], return_counts=True)
                team_final = vals[np.argmax(counts)]
            else:
                team_final = team_raw

            player_info["team_final"] = team_final
            frame_players.append(player_info)

        results.append({
            "frame_index": frame_index,
            "players": frame_players,
        })

    cap.release()

    # ============================
    # Guardar resultados
    # ============================

    output_key = video_key.replace(".mp4", "_teams.json")
    upload_json_to_s3(results, output_key)

    elapsed = time.time() - start_time
    log(f"Finished processing in {elapsed:.1f}s")

    if user_email:
        send_email(
            user_email,
            "Video processed",
            f"Your video {video_key} was processed in {elapsed:.1f} seconds.\nResults: {output_key}",
        )


if __name__ == "__main__":
    try:
        video_key = os.getenv("VIDEO_KEY")
        user_email = os.getenv("USER_EMAIL")

        if not video_key:
            raise RuntimeError("VIDEO_KEY not set")

        process_video(video_key, user_email)
    except Exception as e:
        log("Error in process_video.py:")
        log(str(e))
        log(traceback.format_exc())
