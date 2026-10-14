# ============================
# Toggles
# ============================
USE_CLUSTERING = True  # ativa clusterização por cores (KMeans)

import sys
import os
import cv2
import shutil
import json
import requests
import numpy as np
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process.core.models import load_yolo_model
from process.core.cropper import is_box_too_small
from process.core.ema import init_track_memory, update_ema, decide_team
from process.core.fallback import apply_fallback
from process.core.drawer import draw_entity
from process.io.storage import list_incoming, download_video_and_json, upload_file, delete_incoming
from process.io.ffmpeg import cut_video, convert_video_to_safe_format
from process.io.emailer import send_email

print("\n==============================")
print("   PROCESS VIDEO PIPELINE — clusterização + temporal")
print("==============================\n")

# -----------------------------
# Clusterização por cores (Hue + Saturação)
# -----------------------------
def crop_jersey_only(frame, x1, y1, x2, y2):
    h_frame, w_frame, _ = frame.shape

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    box = frame[y1:y2, x1:x2]
    if box.size == 0:
        return box

    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    mask = (
        (s_channel > 70) &
        (v_channel > 40) &
        (v_channel < 220)
    )

    if mask.sum() < 50:
        h_box, w_box, _ = box.shape
        ny1 = int(h_box * 0.25)
        ny2 = int(h_box * 0.75)
        nx1 = int(w_box * 0.20)
        nx2 = int(w_box * 0.80)
        return box[ny1:ny2, nx1:nx2]

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h_box, w_box, _ = box.shape
        ny1 = int(h_box * 0.25)
        ny2 = int(h_box * 0.75)
        nx1 = int(w_box * 0.20)
        nx2 = int(w_box * 0.80)
        return box[ny1:ny2, nx1:nx2]

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1b = min(box.shape[1], x + w + pad_x)
    y1b = min(box.shape[0], y + h + pad_y)

    jersey_crop = box[y0:y1b, x0:x1b]
    return jersey_crop


def extract_jersey_feature(crop):
    if crop is None or crop.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    mask = (
        (s_channel > 70) &
        (v_channel > 40) &
        (v_channel < 220)
    )

    filtered_hue = h_channel[mask]
    filtered_sat = s_channel[mask]

    if len(filtered_hue) < 50:
        return np.array([0.0, 0.0], dtype=np.float32)

    hist, _ = np.histogram(filtered_hue, bins=180, range=(0, 180))
    dominant_hue = float(np.argmax(hist))
    mean_sat = float(filtered_sat.mean())

    return np.array([dominant_hue, mean_sat], dtype=np.float32)


def cluster_players_kmeans(features):
    if len(features) == 0:
        return []
    if len(features) == 1:
        return [0]

    X = np.stack(features, axis=0)
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)
    return labels.tolist()


# -----------------------------
# PROCESS VIDEO
# -----------------------------
def process_video(model, local_video, original_name):
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

    team_memory = {}
    last_team = {}
    fallback_counter = {}

    cluster_memory = {}  # track_id -> lista de "A"/"B"

    CLASSIFY_CLASSES = ("player", "goalkeeper")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes

        frame_name = f"frame_{frame_index:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)

        players = []
        features = []

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

            if base_label not in CLASSIFY_CLASSES:
                continue

            role = base_label

            if track_id is not None:
                init_track_memory(team_memory, fallback_counter, last_team, track_id)

            if is_box_too_small(w_box, h_box):
                players.append({
                    "role": role,
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "feature": None,
                    "crop": None,
                    "team_raw": None
                })
                continue

            crop = crop_jersey_only(frame, x1, y1, x2, y2)

            players.append({
                "role": role,
                "track_id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop": crop,
                "feature": None,
                "team_raw": None
            })

        # -----------------------------
        # 2) CLUSTERIZAÇÃO
        # -----------------------------
        if USE_CLUSTERING:
            for p in players:
                if p["crop"] is None:
                    continue
                p["feature"] = extract_jersey_feature(p["crop"])
                features.append(p["feature"])

            cluster_labels = cluster_players_kmeans(features)

            idx = 0
            for p in players:
                if p["feature"] is None:
                    p["team_raw"] = None
                    continue

                cluster_id = cluster_labels[idx]
                idx += 1
                p["team_raw"] = "A" if cluster_id == 0 else "B"
        else:
            for p in players:
                p["team_raw"] = None

        # -----------------------------
        # 3) MEMÓRIA TEMPORAL DE CLUSTERIZAÇÃO
        # -----------------------------
        for p in players:
            track_id = p["track_id"]
            team_raw = p["team_raw"]

            if track_id is not None and team_raw is not None:
                if track_id not in cluster_memory:
                    cluster_memory[track_id] = []
                cluster_memory[track_id].append(team_raw)
                if len(cluster_memory[track_id]) > 10:
                    cluster_memory[track_id].pop(0)

        # -----------------------------
        # 4) EMA + FALLBACK + DECISÃO FINAL
        # -----------------------------
        for p in players:
            role = p["role"]
            track_id = p["track_id"]
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            team_raw = p["team_raw"]

            if track_id is not None:
                history = cluster_memory.get(track_id, [])
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
    print("   PROCESS VIDEO PIPELINE — clusterização + temporal")
    print("==============================\n")
    print(">>> USING process_video — clusterização + toggles <<<")

    model = load_yolo_model()

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
            model, safe_video, os.path.basename(key)
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


if __name__ == "__main__":
    main()
