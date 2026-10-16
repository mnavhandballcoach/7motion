import os
import cv2
import json
import shutil
import numpy as np
from collections import Counter
import requests

from process.core.models import load_yolo_model, load_team_classifier
from process.core.cropper import crop_with_pad, is_box_too_small
from process.core.classifier import TeamClassifier
from process.core.drawer import draw_entity

from process.io.storage import (
    list_incoming, download_video_and_json, upload_file, delete_incoming
)
from process.io.ffmpeg import cut_video, convert_video_to_safe_format
from process.io.emailer import send_email


# ============================================================
# UPLOAD DE CROPS PARA BACKBLAZE
# ============================================================
def upload_debug_crop(local_path, video_id, frame_index, filename):
    remote_path = f"processed/crops/{video_id}/frame_{frame_index:05d}/{filename}"
    upload_file(local_path, remote_path)


# ============================================================
# GUARDAR + ENVIAR CROP
# ============================================================
def save_debug_crop(crop, frame_index, track_id, cls_name=None, conf=None, video_id=None):
    if crop is None:
        return

    debug_dir = os.path.join("debug_crops", f"frame_{frame_index:05d}")
    os.makedirs(debug_dir, exist_ok=True)

    if cls_name is not None:
        fname = f"track_{track_id}_{cls_name}_{conf:.2f}.jpg"
    else:
        fname = f"track_{track_id}.jpg"

    local_path = os.path.join(debug_dir, fname)
    cv2.imwrite(local_path, crop)

    if video_id is not None:
        upload_debug_crop(local_path, video_id, frame_index, fname)


# ============================================================
# PROCESSAMENTO DO VÍDEO (SEM EMA, SEM HEURÍSTICAS)
# ============================================================
def process_video(model, classifier, local_video, original_name):

    cap = cv2.VideoCapture(local_video)
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
    CLASSIFY_CLASSES = ("player",)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes

        frame_name = f"frame_{frame_index:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)

        players = []

        # ============================================================
        # DETEÇÃO
        # ============================================================
        for box in detections:
            cls = int(box.cls[0])
            base_label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1
            track_id = int(box.id[0]) if box.id is not None else None

            # árbitro
            if base_label == "referee":
                label_str = draw_entity(frame, "referee", None, x1, y1, x2, y2)
                labels_file.write(f"{frame_name} {label_str}\n")
                continue

            # bola
            if base_label == "ball":
                label_str = draw_entity(frame, "ball", None, x1, y1, x2, y2)
                labels_file.write(f"{frame_name} {label_str}\n")
                continue

            # guarda-redes
            if base_label == "goalkeeper":
                crop, _ = crop_with_pad(frame, x1, y1, x2, y2, w, h)
                save_debug_crop(crop, frame_index, track_id, "goalkeeper", 1.0, base_name)

                players.append({
                    "role": "goalkeeper",
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "crop": crop,
                    "cls_name": None,
                    "conf": None,
                })
                continue

            # só jogadores
            if base_label not in CLASSIFY_CLASSES:
                continue

            if is_box_too_small(w_box, h_box):
                players.append({
                    "role": "player",
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "crop": None,
                    "cls_name": None,
                    "conf": None,
                })
                continue

            crop, _ = crop_with_pad(frame, x1, y1, x2, y2, w, h)
            save_debug_crop(crop, frame_index, track_id, "raw", 0.0, base_name)

            players.append({
                "role": "player",
                "track_id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop": crop,
                "cls_name": None,
                "conf": None,
            })

        # ============================================================
        # CLASSIFICADOR (CRU)
        # ============================================================
        if classifier is not None:
            for p in players:
                if p["role"] != "player":
                    continue

                crop = p["crop"]
                if crop is None:
                    continue

                cls_name, conf = classifier.predict_with_confidence(crop)
                p["cls_name"] = cls_name
                p["conf"] = conf

                save_debug_crop(crop, frame_index, p["track_id"], cls_name, conf, base_name)

        # ============================================================
        # ANOTAÇÃO CRUA (SEM EQUIPAS)
        # ============================================================
        for p in players:
            role = p["role"]
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            cls_name = p["cls_name"]
            conf = p["conf"]

            label_str = draw_entity(frame, role, cls_name, x1, y1, x2, y2)
            labels_file.write(f"{frame_name} {label_str}\n")

        # guarda-redes sem equipa
        for p in players:
            if p["role"] != "goalkeeper":
                continue

            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            label_str = draw_entity(frame, "goalkeeper", None, x1, y1, x2, y2)
            labels_file.write(f"{frame_name} {label_str}\n")

        cv2.imwrite(frame_path, frame)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    labels_file.close()

    return output_video, frames_dir, labels_path


# ============================================================
# MAIN
# ============================================================
def main():

    model = load_yolo_model()

    classifier_path = load_team_classifier()
    classifier = TeamClassifier(classifier_path)

    files_all, files_videos = list_incoming()
    if not files_videos:
        print("No videos in incoming/.")
        return

    for file in files_videos:
        key = file["Key"]

        local_video, local_json, json_key = download_video_and_json(key)
        delete_incoming(key, json_key)

        with open(local_json, "r") as f:
            data = json.load(f)

        user_email = data.get("email")
        user_name = data.get("name")
        start_time = int(data.get("start", 0))
        duration = int(data.get("duration", 0))

        cut_path = cut_video(local_video, start_time, duration) if duration > 0 else local_video
        safe_video = convert_video_to_safe_format(cut_path)

        output_video, frames_dir, labels_path = process_video(
            model, classifier, safe_video, os.path.basename(key)
        )

        upload_file(output_video, f"processed/{os.path.basename(output_video)}")
        upload_file(local_video, f"processed/originals/{os.path.basename(key)}")

        for root, dirs, files_local in os.walk(frames_dir):
            for f in files_local:
                upload_file(os.path.join(root, f),
                            f"processed/frames/{os.path.basename(frames_dir)}/{f}")

        upload_file(labels_path, f"processed/labels/{os.path.basename(labels_path)}")

        video_url = (
            f"https://f003.backblazeb2.com/file/"
            f"{os.getenv('VIDEOS_BUCKET', 'handball-videos')}/"
            f"processed/{os.path.basename(output_video)}"
        )

        wordpress_endpoint = "https://mariocesar.navarro.pt/wp/wp-json/7motion/v1/create-video-page"
        payload = {
            "video_url": video_url,
            "user_email": user_email,
            "user_name": user_name,
            "video_id": os.path.splitext(os.path.basename(key))[0],
        }

        try:
            resp = requests.post(wordpress_endpoint, json=payload, timeout=20)
            page_url = resp.json().get("page_url", video_url)
        except Exception:
            page_url = video_url

        send_email(user_email, user_name, page_url)

        shutil.rmtree("output", ignore_errors=True)
        shutil.rmtree("output_frames", ignore_errors=True)
        shutil.rmtree("output_labels", ignore_errors=True)

        os.remove(local_video)
        os.remove(safe_video)
        os.remove(local_json)


if __name__ == "__main__":
    main()
