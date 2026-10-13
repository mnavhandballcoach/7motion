# ============================
# Toggles
# ============================
USE_TEAM_CLASSIFIER = True       # usa o teu classificador de equipas
USE_CLUSTERING = True            # usa clustering por cores
N_CALIB_FRAMES = 200             # nº de frames para calibração inicial
JERSEY_TOP = 0.2                 # percentagem topo do tronco
JERSEY_BOTTOM = 0.6              # percentagem fundo do tronco

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

from process.core.models import load_yolo_model, load_team_classifier
from process.core.cropper import crop_with_pad, is_box_too_small
from process.core.classifier import TeamClassifier
from process.core.ema import init_track_memory, update_ema, decide_team
from process.core.fallback import apply_fallback
from process.core.drawer import draw_entity
from process.core.utils import is_player_class

from process.io.storage import list_incoming, download_video_and_json, upload_file, delete_incoming
from process.io.ffmpeg import cut_video, convert_video_to_safe_format
from process.io.emailer import send_email

print("\n==============================")
print("   PROCESS VIDEO PIPELINE — calibração + clustering + tracking")
print("==============================\n")


# -----------------------------
# 1) Recorte da camisola
# -----------------------------
def crop_jersey(crop):
    h, w, _ = crop.shape
    y1 = int(h * JERSEY_TOP)
    y2 = int(h * JERSEY_BOTTOM)
    return crop[y1:y2, :]


# -----------------------------
# 2) Histograma HSV (Hue + Saturation)
# -----------------------------
def extract_hsv_histogram(crop, bins_h=16, bins_s=8):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hist_h = cv2.calcHist([h], [0], None, [bins_h], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [bins_s], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()

    feature = np.concatenate([hist_h, hist_s]).astype(np.float32)
    return feature


# -----------------------------
# 3) Clustering k=4 na calibração
# -----------------------------
def cluster_calibration(features, n_clusters=4):
    if len(features) < n_clusters:
        return None, None

    X = np.stack(features, axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    return labels, centers


def assign_cluster(feature, centers):
    dists = np.linalg.norm(centers - feature, axis=1)
    return int(np.argmin(dists))


# -----------------------------
# 4) Definir quais clusters são equipas A/B
# -----------------------------
def identify_team_clusters(features, labels):
    # conta elementos por cluster
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1

    # ordena clusters por tamanho (desc)
    sorted_clusters = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_clusters) < 2:
        return None, None

    team_cluster_A = sorted_clusters[0][0]
    team_cluster_B = sorted_clusters[1][0]

    return team_cluster_A, team_cluster_B


# -----------------------------
# PROCESS VIDEO
# -----------------------------
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

    team_memory = {}
    last_team = {}
    fallback_counter = {}

    # memória temporal por track_id
    track_votes = {}       # track_id -> lista de "A"/"B"
    cluster_centers = None
    team_cluster_A = None
    team_cluster_B = None
    calib_features = []

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
                    "jersey_crop": None,
                    "feature": None,
                    "team_raw": None
                })
                continue

            crop, _ = crop_with_pad(frame, x1, y1, x2, y2, w, h)
            jersey = crop_jersey(crop)

            players.append({
                "role": role,
                "track_id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop": crop,
                "jersey_crop": jersey,
                "feature": None,
                "team_raw": None
            })

        # -----------------------------
        # 2) CALIBRAÇÃO INICIAL (primeiros N_CALIB_FRAMES)
        # -----------------------------
        if frame_index < N_CALIB_FRAMES and USE_CLUSTERING:
            for p in players:
                if p["jersey_crop"] is None:
                    continue
                feat = extract_hsv_histogram(p["jersey_crop"])
                calib_features.append(feat)

            if frame_index == N_CALIB_FRAMES - 1 and len(calib_features) >= 8:
                labels, centers = cluster_calibration(calib_features, n_clusters=4)
                if centers is not None:
                    cluster_centers = centers
                    team_cluster_A, team_cluster_B = identify_team_clusters(calib_features, labels)
                    print("Calibração concluída:")
                    print("  team_cluster_A:", team_cluster_A)
                    print("  team_cluster_B:", team_cluster_B)

        # -----------------------------
        # 3) ATRIBUIÇÃO DE EQUIPA (após calibração)
        # -----------------------------
        for p in players:
            track_id = p["track_id"]
            jersey = p["jersey_crop"]

            if jersey is None:
                p["team_raw"] = None
                continue

            # feature para clustering
            feat = extract_hsv_histogram(jersey)
            p["feature"] = feat

            team_kmeans = None
            if USE_CLUSTERING and cluster_centers is not None and team_cluster_A is not None and team_cluster_B is not None:
                cluster_id = assign_cluster(feat, cluster_centers)
                if cluster_id == team_cluster_A:
                    team_kmeans = "A"
                elif cluster_id == team_cluster_B:
                    team_kmeans = "B"
                else:
                    team_kmeans = None

            team_cls = None
            conf_cls = 0.0
            if USE_TEAM_CLASSIFIER and p["crop"] is not None:
                team_cls, conf_cls = classifier.predict_with_confidence(p["crop"])

            # fusão simples: se classificador confiante, usa; senão KMeans
            if team_cls is not None and conf_cls >= 0.7:
                team_raw = team_cls
            elif team_kmeans is not None:
                team_raw = team_kmeans
            else:
                team_raw = team_cls if team_cls is not None else None

            p["team_raw"] = team_raw

            # votos por track_id
            if track_id is not None and team_raw is not None:
                if track_id not in track_votes:
                    track_votes[track_id] = []
                track_votes[track_id].append(team_raw)
                if len(track_votes[track_id]) > 50:
                    track_votes[track_id].pop(0)

        # -----------------------------
        # 4) EMA + FALLBACK + MODA POR TRACK_ID
        # -----------------------------
        for p in players:
            role = p["role"]
            track_id = p["track_id"]
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            team_raw = p["team_raw"]

            # moda por track_id
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
    print("   PROCESS VIDEO PIPELINE — calibração + clustering + tracking")
    print("==============================\n")

    model = load_yolo_model()

    classifier = None
    if USE_TEAM_CLASSIFIER:
        load_team_classifier()
        classifier = TeamClassifier("bestclassifier.pt")

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
