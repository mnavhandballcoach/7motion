# ============================
# Toggles
# ============================
USE_CLUSTERING = True
N_CALIB_FRAMES = 200  # nº de frames para calibrar equipas
JERSEY_TOP = 0.2
JERSEY_BOTTOM = 0.6

import sys
import os
import cv2
import shutil
import json
import requests
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process.core.models import load_yolo_model
from process.core.cropper import crop_with_pad, is_box_too_small
from process.core.ema import init_track_memory, update_ema, decide_team
from process.core.fallback import apply_fallback
from process.core.drawer import draw_entity
from process.core.utils import is_player_class

# === CLUSTERING UNIVERSAL ===
from process.core.clustering import (
    crop_jersey_only,
    extract_embedding,
    run_initial_clustering,
    identify_team_clusters,
    decide_team_A_B,
    assign_cluster,
    assign_team
)

from process.io.storage import list_incoming, download_video_and_json, upload_file, delete_incoming
from process.io.ffmpeg import cut_video, convert_video_to_safe_format
from process.io.emailer import send_email

print("\n==============================")
print("   PROCESS VIDEO PIPELINE — clustering universal")
print("==============================\n")


# -----------------------------
# PROCESS VIDEO
# -----------------------------
def process_video(model, classifier_unused, local_video, original_name):
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

    # === CLUSTERING STATE ===
    embeddings = []
    bboxes = []
    centroids = None
    team_A = None
    team_B = None

    # === TRACKING STATE ===
    team_memory = {}
    last_team = {}
    fallback_counter = {}
    track_votes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes

        frame_name = f"frame_{frame_index:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)

        players = []

        # -----------------------------
        # 1) DETEÇÃO
        # -----------------------------
        for box in detections:
            cls = int(box.cls[0])
            base_label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1

            track_id = int(box.id[0]) if box.id is not None else None

            if base_label == "referee":
                label_str = draw_entity(frame, "referee", None, x1, y1, x2, y2)
                labels_file.write(f"{frame_name} {label_str}\n")
                continue

            if base_label == "ball":
                label_str = draw_entity(frame, "ball", None, x1, y1, x2, y2)
                labels_file.write(f"{frame_name} {label_str}\n")
                continue

            if not is_player_class(base_label):
                continue

            role = base_label

            if track_id is not None:
                init_track_memory(team_memory, fallback_counter, last_team, track_id)

            if is_box_too_small(w_box, h_box):
                players.append({
                    "role": role,
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "crop": None,
                    "team_raw": None
                })
                continue

            crop, _ = crop_with_pad(frame, x1, y1, x2, y2, w, h)

            players.append({
                "role": role,
                "track_id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop": crop,
                "team_raw": None
            })

        # -----------------------------
        # 2) CLUSTERING UNIVERSAL
        # -----------------------------
        for p in players:
            crop = p["crop"]
            track_id = p["track_id"]

            if crop is None:
                p["team_raw"] = None
                continue

            # === CALIBRAÇÃO INICIAL ===
            if frame_index < N_CALIB_FRAMES:
                jersey = crop_jersey_only(crop)
                emb = extract_embedding(jersey)
                embeddings.append(emb)
                bboxes.append((p["x1"], p["y1"], p["x2"], p["y2"]))
                p["team_raw"] = None
                continue

            # === CORRER CLUSTERING NO FRAME EXACTO ===
            if frame_index == N_CALIB_FRAMES:
                centroids, labels = run_initial_clustering(embeddings)
                team_clusters = identify_team_clusters(labels)
                team_A, team_B = decide_team_A_B(team_clusters, bboxes, labels)

                print("=== CLUSTERING CALIBRADO ===")
                print("Team A cluster:", team_A)
                print("Team B cluster:", team_B)

            # === ATRIBUIÇÃO POR CENTRÓIDE ===
            jersey = crop_jersey_only(crop)
            emb = extract_embedding(jersey)
            cluster_id = assign_cluster(emb, centroids)
            team_raw = assign_team(cluster_id, team_A, team_B)

            p["team_raw"] = team_raw

            # === TRACK VOTES ===
            if track_id is not None:
                if track_id not in track_votes:
                    track_votes[track_id] = []
                track_votes[track_id].append(team_raw)
                if len(track_votes[track_id]) > 50:
                    track_votes[track_id].pop(0)

        # -----------------------------
        # 3) EMA + FALLBACK + TRACKING
        # -----------------------------
        for p in players:
            role = p["role"]
            track_id = p["track_id"]
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            team_raw = p["team_raw"]

            if track_id is not None:
                history = track_votes.get(track_id, [])
                if history:
                    team_temporal = max(set(history), key=history.count)
                else:
                    team_temporal = team_raw
            else:
                team_temporal = team_raw

            if track_id is not None:
                update_ema(team_memory, track_id, team_temporal)
                team = decide_team(team_memory, track_id)
                team = apply_fallback(track_id, team, last_team, fallback_counter)
            else:
                team = team_temporal

            label_str = draw_entity(frame, role, team, x1, y1, x2, y2)
            labels_file.write(f"{frame_name} {label_str}\n")

        cv2.imwrite(frame_path, frame)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    labels_file.close()

    return output_video, frames_dir, labels_path


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("==============================")
    print("   PROCESS VIDEO PIPELINE — clustering universal")
    print("==============================\n")

    model = load_yolo_model()
    classifier = None  # clustering substitui classificador

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

        video_url = (
            f"https://f003.backblazeb2.com/file/"
            f"{os.getenv('VIDEOS_BUCKET', 'handball-videos')}/"
            f"{annotated_remote}"
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
            data_wp = resp.json()
            page_url = data_wp.get("page_url", video_url)
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
