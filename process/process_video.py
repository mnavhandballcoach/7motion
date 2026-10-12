import os
import cv2
import shutil
import boto3
import requests
import subprocess
import numpy as np
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO

from api.b2_client import hard_delete_file

print("\n==============================")
print("   PROCESS VIDEO PIPELINE v13")
print("==============================\n")

USE_TEAM_CLASSIFIER = True

MODEL_URL = "https://github.com/mnavhandballcoach/7motion/releases/download/models/best.pt"
CLASSIFIER_URL = "https://github.com/mnavhandballcoach/7motion/releases/download/models/bestclassifier.pt"

BUCKET_NAME = "handball-videos"
S3_ENDPOINT = "https://s3.eu-central-003.backblazeb2.com"

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET
)

def send_email(user_email, user_name, video_url):
    if not user_email:
        print("⚠️ No email provided, skipping email notification.")
        return

    subject = "O seu vídeo anotado está pronto!"
    message = f"""
Olá {user_name or "Utilizador"},

O seu vídeo foi processado com sucesso.

Pode descarregar o vídeo anotado aqui:
{video_url}

Obrigado por utilizar o nosso serviço!
"""

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = user_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, user_email, msg.as_string())
        print("✔ Email enviado\n")
    except Exception as e:
        print("❌ Erro ao enviar email:", e)


def download_model(url, filename):
    print(f"⬇️ Downloading model: {filename}")
    r = requests.get(url)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"✔ Model downloaded: {filename}\n")
    return filename


def list_incoming():
    print("📂 Listing videos in bucket (incoming/)...")
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="incoming/")
    files = resp.get("Contents", []) or []

    video_exts = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in files if f["Key"].lower().endswith(video_exts)]

    print(f"✔ Found {len(videos)} videos in incoming/\n")
    return files, videos


def download_video_and_json(key):
    print(f"⬇️ Downloading video + JSON for {key}")

    local_video = "input.mp4"
    local_json = "input.json"

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    with open(local_video, "wb") as f:
        f.write(obj["Body"].read())
    print("✔ Video downloaded")

    json_key = key.replace(".mp4", ".json")
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=json_key)
        with open(local_json, "wb") as f:
            f.write(obj["Body"].read())
        print("✔ JSON downloaded")
    except Exception:
        print("⚠️ JSON not found, using defaults")
        with open(local_json, "w") as f:
            json.dump({"email": None, "name": None, "start": 0, "duration": 0}, f)

    print("✔ Download complete\n")
    return local_video, local_json


def cut_video(input_path, start_time, duration):
    print(f"✂️ Cutting video: start={start_time}, duration={duration}")
    output_path = "cut_input.mp4"
    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", input_path, "-t", str(duration), "-c", "copy", output_path]
    subprocess.run(cmd, check=True)
    print("✔ Cut complete\n")
    return output_path


def convert_video_to_safe_format(input_path):
    print("🎞 Converting video to safe format...")
    safe_path = "safe_input.mp4"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-movflags", "faststart", "-pix_fmt", "yuv420p", safe_path]
    subprocess.run(cmd, check=True)
    print("✔ Safe video created:", safe_path, "\n")
    return safe_path


def upload_file(local_path, remote_name):
    print(f"⬆️ Uploading: {remote_name}")
    s3.upload_file(local_path, BUCKET_NAME, remote_name)
    print("✔ Upload complete\n")


COLOR_PLAYER_YOLO = (180, 80, 80)
COLOR_GOALKEEPER_YOLO = (120, 220, 120)
COLOR_REFEREE = (200, 200, 200)
COLOR_BALL = (255, 255, 0)

COLOR_TEAM_A = (0, 0, 255)
COLOR_TEAM_B = (0, 255, 0)

def draw_triangle(frame, x, y, color):
    pts = np.array([[x, y], [x - 14, y - 22], [x + 14, y - 22]], np.int32)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
    cv2.fillPoly(frame, [pts], color)

def draw_label(frame, x, y, label):
    cv2.putText(frame, label, (x - 35, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, label, (x - 35, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def draw_ball(frame, x, y, w, h):
    radius = int(max(w, h) / 2)
    cx = x + w // 2
    cy = y + h // 2
    cv2.circle(frame, (cx, cy), radius, (0, 0, 0), 3)
    cv2.circle(frame, (cx, cy), radius, COLOR_BALL, 2)


def parse_team_classifier_label(label_str):
    label_str = label_str.strip().lower()
    team = None
    role = None

    if "team a" in label_str:
        team = "A"
    elif "team b" in label_str:
        team = "B"

    if "player" in label_str:
        role = "player"
    elif "goalkeeper" in label_str:
        role = "goalkeeper"

    return role, team


def process_video(model, classifier, local_video, original_name):
    print(f"🔍 Running YOLO (tracking) on {local_video}...")

    cap = cv2.VideoCapture(local_video)
    if not cap.isOpened():
        raise RuntimeError("❌ OpenCV cannot open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(original_name)[0]

    stable_dir = "output"
    os.makedirs(stable_dir, exist_ok=True)
    output_video = os.path.join(stable_dir, f"{base_name}_annotated.mp4")

    frames_dir = os.path.join("output_frames", base_name)
    labels_dir = "output_labels"
    labels_path = os.path.join(labels_dir, f"{base_name}.txt")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    labels_file = open(labels_path, "w")

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (w, h)
    )

    frame_index = 0

    team_memory = {}

    CONF_THRESHOLD = 0.4
    EMA_ALPHA = 0.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        detections = results[0].boxes

        frame_name = f"frame_{frame_index:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)

        for box in detections:
            cls = int(box.cls[0])
            base_label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            final_label = base_label
            color = None

            if base_label == "ball":
                draw_ball(frame, x1, y1, w_box, h_box)
                final_label = "Ball"
                color = COLOR_BALL

            elif base_label == "referee":
                draw_triangle(frame, x1 + w_box // 2, y1, COLOR_REFEREE)
                final_label = "Referee"
                color = COLOR_REFEREE

            else:
                if USE_TEAM_CLASSIFIER and classifier is not None and track_id is not None:
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                    if crop.size > 0:
                        cls_results = classifier.predict(crop, verbose=False)
                        cls_boxes = cls_results[0].boxes

                        # PATCH: nunca sair antes de desenhar
                        if cls_boxes is None or len(cls_boxes) == 0:
                            if base_label == "player":
                                final_label = "Player"
                                color = COLOR_PLAYER_YOLO
                            else:
                                final_label = "Goalkeeper"
                                color = COLOR_GOALKEEPER_YOLO

                            draw_triangle(frame, x1 + w_box // 2, y1, color)
                            draw_label(frame, x1 + w_box // 2, y1, final_label)

                            labels_file.write(
                                f"{frame_name} {final_label} - {x1} {y1} {x2} {y2}\n"
                            )
                            continue

                        cbox = cls_boxes[0]
                        ccls = int(cbox.cls[0])
                        raw_label = classifier.names[ccls]
                        role, team = parse_team_classifier_label(raw_label)

                        conf = float(cbox.conf[0]) if cbox.conf is not None else 1.0

                        if role in ("player", "goalkeeper") and team in ("A", "B"):

                            prev = team_memory.get(track_id)

                            if prev is None:
                                team_memory[track_id] = {
                                    "team": team,
                                    "role": role,
                                    "confidence": conf
                                }
                            else:
                                if prev["team"] == team:
                                    new_conf = EMA_ALPHA * conf + (1 - EMA_ALPHA) * prev["confidence"]
                                    team_memory[track_id]["confidence"] = new_conf
                                else:
                                    if conf > prev["confidence"] + 0.2:
                                        team_memory[track_id] = {
                                            "team": team,
                                            "role": role,
                                            "confidence": conf
                                        }

                            mem = team_memory[track_id]

                            if mem["confidence"] >= CONF_THRESHOLD:
                                if mem["role"] == "player":
                                    final_label = f"Player Team {mem['team']}"
                                    color = COLOR_TEAM_A if mem["team"] == "A" else COLOR_TEAM_B
                                else:
                                    final_label = f"Goalkeeper Team {mem['team']}"
                                    color = COLOR_TEAM_A if mem["team"] == "A" else COLOR_TEAM_B
                            else:
                                if base_label == "player":
                                    final_label = "Player"
                                    color = COLOR_PLAYER_YOLO
                                else:
                                    final_label = "Goalkeeper"
                                    color = COLOR_GOALKEEPER_YOLO

                        else:
                            if base_label == "player":
                                final_label = "Player"
                                color = COLOR_PLAYER_YOLO
                            else:
                                final_label = "Goalkeeper"
                                color = COLOR_GOALKEEPER_YOLO

                    else:
                        if base_label == "player":
                            final_label = "Player"
                            color = COLOR_PLAYER_YOLO
                        else:
                            final_label = "Goalkeeper"
                            color = COLOR_GOALKEEPER_YOLO

                else:
                    if base_label == "player":
                        final_label = "Player"
                        color = COLOR_PLAYER_YOLO
                    else:
                        final_label = "Goalkeeper"
                        color = COLOR_GOALKEEPER_YOLO

                draw_triangle(frame, x1 + w_box // 2, y1, color)
                draw_label(frame, x1 + w_box // 2, y1, final_label)

            labels_file.write(
                f"{frame_name} {final_label} - {x1} {y1} {x2} {y2}\n"
            )

        cv2.imwrite(frame_path, frame)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    labels_file.close()

    print(f"✔ Annotated video created: {output_video}")
    print(f"✔ Frames saved in: {frames_dir}")
    print(f"✔ Labels saved in: {labels_path}\n")

    return output_video, frames_dir, labels_path


def main():
    print("🚀 Starting pipeline...\n")

    model_path = download_model(MODEL_URL, "best.pt")
    model = YOLO(model_path)

    classifier = None
    if USE_TEAM_CLASSIFIER:
        classifier_path = download_model(CLASSIFIER_URL, "bestclassifier.pt")
        classifier = YOLO(classifier_path)

    files_all, files_videos = list_incoming()
    if not files_videos:
        print("📭 No videos to process.\n")
        return

    for file in files_videos:
        key = file["Key"]
        json_key = key.replace(".mp4", ".json")

        print("\n========================================")
        print(f"🎬 PROCESSING VIDEO: {key}")
        print("========================================\n")

        local_video, local_json = download_video_and_json(key)

        print(f"🗑️ HARD DELETE incoming video: {key}")
        hard_delete_file(key)

        print(f"🗑️ HARD DELETE incoming JSON: {json_key}")
        hard_delete_file(json_key)

        with open(local_json, "r") as f:
            data = json.load(f)

        user_email = data.get("email")
        user_name = data.get("name")
        start_time = int(data.get("start", 0))
        duration = int(data.get("duration", 0))

        if duration > 0:
            cut_path = cut_video(local_video, start_time, duration)
        else:
            cut_path = local_video

        safe_video = convert_video_to_safe_format(cut_path)

        output_video, frames_dir, labels_path = process_video(
            model, classifier, safe_video, os.path.basename(key)
        )

        annotated_remote = f"processed/{os.path.basename(output_video)}"
        upload_file(output_video, annotated_remote)

        original_remote = f"processed/originals/{os.path.basename(key)}"
        upload_file(local_video, original_remote)

        for root, dirs, files_local in os.walk(frames_dir):
            for f in files_local:
                local_path = os.path.join(root, f)
                remote_path = f"processed/frames/{os.path.basename(frames_dir)}/{f}"
                upload_file(local_path, remote_path)

        upload_file(labels_path, f"processed/labels/{os.path.basename(labels_path)}")

        video_url = f"https://f003.backblazeb2.com/file/{BUCKET_NAME}/{annotated_remote}"
        send_email(user_email, user_name, video_url)

        shutil.rmtree("output", ignore_errors=True)
        shutil.rmtree("output_frames", ignore_errors=True)
        shutil.rmtree("output_labels", ignore_errors=True)

        os.remove(local_video)
        os.remove(safe_video)
        os.remove(local_json)

        print("✔ Cleanup complete for this video\n")

    print("🎉 All videos processed.\n")


if __name__ == "__main__":
    main()
