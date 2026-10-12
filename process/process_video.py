import os
import cv2
import shutil
import boto3
import requests
import subprocess
import numpy as np
import json
from ultralytics import YOLO

from api.b2_client import hard_delete_file

# =============================================================
# DEBUG FLAGS
# =============================================================
DEBUG = True
DEBUG_CROP = True
DEBUG_CLASSIFIER = True
DEBUG_TEAM = True
DEBUG_TRACK = True

# =============================================================
# CONFIG
# =============================================================
USE_TEAM_CLASSIFIER = True

MODEL_URL = "https://github.com/mnavhandballcoach/7motion/releases/download/models/best.pt"
CLASSIFIER_URL = "https://github.com/mnavhandballcoach/7motion/releases/download/models/bestclassifier.pt"

BUCKET_NAME = "handball-videos"
S3_ENDPOINT = "https://s3.eu-central-003.backblazeb2.com"

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET
)

# =============================================================
# HELPERS
# =============================================================

def parse_team_classifier_label(label_str):
    label_str = label_str.strip().lower()
    if "team a" in label_str:
        return "A"
    if "team b" in label_str:
        return "B"
    return None


def download_model(url, filename):
    print(f"⬇️ Downloading model: {filename}")
    r = requests.get(url)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"✔ Model downloaded: {filename}\n")
    return filename


def list_incoming():
    print("📂 Listing videos in incoming/")
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="incoming/")
    files = resp.get("Contents", []) or []
    videos = [f for f in files if f["Key"].lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    print(f"✔ Found {len(videos)} videos\n")
    return videos


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

    return local_video, local_json


def convert_video_to_safe_format(input_path):
    print("🎞 Converting video to safe format...")
    safe_path = "safe_input.mp4"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-movflags", "faststart", "-pix_fmt", "yuv420p", safe_path]
    subprocess.run(cmd, check=True)
    print("✔ Safe video created\n")
    return safe_path

# =============================================================
# DEBUG PROCESSING
# =============================================================

def process_video_debug(model, classifier, local_video, original_name):
    print("\n==============================")
    print("   DEBUG MODE: TeamClassifier")
    print("==============================\n")

    cap = cv2.VideoCapture(local_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "debug_output.mp4"
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (w, h)
    )

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n--- FRAME {frame_index} ---")

        results = model.track(frame, persist=True)
        detections = results[0].boxes

        for box in detections:
            cls = int(box.cls[0])
            base_label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1

            track_id = int(box.id[0]) if box.id is not None else None

            print(f"Object: {base_label} | track_id={track_id}")
            print(f"Box: {x1},{y1},{x2},{y2}")

            # REFEREE / BALL
            if base_label in ["referee", "ball"]:
                print(f"→ YOLO only: {base_label}")
                continue

            # PLAYER / GOALKEEPER
            role = base_label
            team = None

            if classifier is not None:
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                if DEBUG_CROP:
                    print(f"Crop shape: {crop.shape}, empty={crop.size == 0}")

                if crop.size == 0:
                    print("⚠️ Crop vazio → classificador ignorado")
                else:
                    cls_results = classifier.predict(crop, verbose=False)
                    cls_boxes = cls_results[0].boxes

                    print("Classifier raw results:", cls_results)

                    if cls_boxes is None or len(cls_boxes) == 0:
                        print("⚠️ Classifier devolveu 0 boxes")
                    else:
                        cbox = cls_boxes[0]
                        ccls = int(cbox.cls[0])
                        raw_label = classifier.names[ccls]
                        conf = float(cbox.conf[0]) if cbox.conf is not None else 1.0

                        print(f"Classifier label: {raw_label} | conf={conf}")

                        team_detected = parse_team_classifier_label(raw_label)
                        print(f"Team detected: {team_detected}")

                        team = team_detected

            # FINAL LABEL
            if team == "A":
                print("✔ FINAL: Team A")
            elif team == "B":
                print("✔ FINAL: Team B")
            else:
                print("❌ FINAL: YOLO fallback (sem equipa)")

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print("\n==============================")
    print("   DEBUG COMPLETO")
    print("==============================\n")

    return output_path

# =============================================================
# MAIN
# =============================================================

def main():
    print("🚀 Starting DEBUG pipeline...\n")

    model_path = download_model(MODEL_URL, "best.pt")
    model = YOLO(model_path)

    classifier = None
    if USE_TEAM_CLASSIFIER:
        classifier_path = download_model(CLASSIFIER_URL, "bestclassifier.pt")
        classifier = YOLO(classifier_path)
        print("Classifier names:", classifier.names)

    videos = list_incoming()
    if not videos:
        print("📭 No videos in incoming.\n")
        return

    for file in videos:
        key = file["Key"]
        json_key = key.replace(".mp4", ".json")

        print("\n========================================")
        print(f"🎬 DEBUGGING VIDEO: {key}")
        print("========================================\n")

        local_video, local_json = download_video_and_json(key)

        print(f"🗑️ HARD DELETE incoming video: {key}")
        hard_delete_file(key)

        print(f"🗑️ HARD DELETE incoming JSON: {json_key}")
        hard_delete_file(json_key)

        safe_video = convert_video_to_safe_format(local_video)

        debug_output = process_video_debug(model, classifier, safe_video, os.path.basename(key))

        print(f"✔ DEBUG OUTPUT CREATED: {debug_output}\n")

if __name__ == "__main__":
    main()