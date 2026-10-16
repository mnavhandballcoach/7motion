import os
import cv2
import json
import shutil
import numpy as np
import requests

from process.core.models import load_yolo_model, load_team_classifier
from process.core.cropper import (
    crop_torso_advanced,
    is_player_far,
    is_player_near
)
from process.core.classifier import TeamClassifier
from process.core.drawer import draw_entity

from process.io.storage import (
    list_incoming, download_video_and_json, upload_file, delete_incoming
)
from process.io.ffmpeg import cut_video, convert_video_to_safe_format
from process.io.emailer import send_email


# ============================================================
# CONFIGURAÇÃO
# ============================================================
TRACKER = "bytetrack.yaml"

MIN_TEAM_MASS = float(os.getenv("TEAM_MIN_MASS", "0.10"))
MIN_TRACK_OBS = int(os.getenv("TEAM_MIN_TRACK_OBS", "5"))

EXCLUDE_CLASSES = {
    c.strip().lower()
    for c in os.getenv("TEAM_EXCLUDE_CLASSES", "").split(",")
    if c.strip()
}

DEBUG_CROPS = os.getenv("DEBUG_CROPS", "0") == "1"
DEBUG_EVERY = int(os.getenv("DEBUG_CROPS_EVERY", "30"))


# ============================================================
# VOTAÇÃO POR TRACK_ID
# ============================================================
class TrackVotes:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.acc = {}   # track_id -> vetor acumulado
        self.obs = {}   # track_id -> nº de observações

    def update(self, track_id, probs, weight):
        if track_id not in self.acc:
            self.acc[track_id] = np.zeros(self.n_classes, dtype=np.float64)
            self.obs[track_id] = 0
        self.acc[track_id] += probs * weight
        self.obs[track_id] += 1


def choose_kits(votes, names):
    n = votes.n_classes
    excluded = [i for i in range(n) if str(names[i]).lower() in EXCLUDE_CLASSES]

    def scores(min_obs):
        s = np.zeros(n, dtype=np.float64)
        for tid, v in votes.acc.items():
            if votes.obs[tid] < min_obs:
                continue
            total = v.sum()
            if total <= 0:
                continue
            s += v / total
        s[excluded] = 0.0
        return s

    s = scores(MIN_TRACK_OBS)
    if s.sum() <= 0:
        s = scores(1)

    order = np.argsort(-s)
    kits = [int(i) for i in order[:2] if s[i] > 0]
    return kits, s


def team_from_votes(vec, kits):
    if vec is None or len(kits) == 0:
        return None

    total = float(np.sum(vec))
    if total <= 0:
        return None

    score_a = float(vec[kits[0]])
    score_b = float(vec[kits[1]]) if len(kits) > 1 else 0.0

    if (score_a + score_b) / total < MIN_TEAM_MASS:
        return None

    return "A" if score_a >= score_b else "B"
# ============================================================
# PASSAGEM 1: DETEÇÃO + TRACKING + VOTAÇÃO (sem desenhar)
# ============================================================
def analyse_video(model, classifier, local_video, base_name):
    cap = cv2.VideoCapture(local_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    votes = TrackVotes(classifier.n_classes) if classifier is not None else None
    frame_dets = []          # deteções por frame
    original_frames = []     # frames originais (não anotados)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # guardar frame original para a passagem 2
        original_frames.append(frame.copy())

        # tracking YOLO + ByteTrack
        results = model.track(
            frame,
            persist=frame_index > 0,
            tracker=TRACKER,
            verbose=False,
        )

        dets = []
        players = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            base_label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1
            track_id = int(box.id[0]) if box.id is not None else None

            # só classes relevantes
            if base_label not in ("referee", "ball", "goalkeeper", "player"):
                continue

            entry = {
                "label": base_label,
                "track_id": track_id,
                "box": (x1, y1, x2, y2),
                "box_h": h_box,
                "crop": None,
                "frame_probs": None,
            }
            dets.append(entry)

            # só jogadores vão para o classifier
            if base_label == "player":

                # jogador longe → tracking sim, classificação não
                if is_player_far(x1, y1, x2, y2, w, h):
                    players.append(entry)
                    continue

                # jogador perto → crop perfeito (torso)
                entry["crop"] = crop_torso_advanced(frame, x1, y1, x2, y2, w, h)
                players.append(entry)

        # classificação em lote
        if classifier is not None:
            batch = [p for p in players if p["crop"] is not None]
            if batch:
                probs = classifier.predict_batch([p["crop"] for p in batch])

                for p, pr in zip(batch, probs):
                    if p["track_id"] is not None:
                        weight = p["box_h"] / float(h)   # mais perto = mais peso
                        votes.update(p["track_id"], pr, weight)
                    else:
                        p["frame_probs"] = pr

                    # crops de debug (opcional)
                    if DEBUG_CROPS and frame_index % DEBUG_EVERY == 0:
                        idx = int(pr.argmax())
                        save_debug_crop(
                            p["crop"], frame_index, p["track_id"],
                            classifier.names[idx], float(pr[idx]), base_name
                        )

        # libertar memória
        for e in dets:
            e["crop"] = None

        frame_dets.append(dets)
        frame_index += 1

    cap.release()
    return frame_dets, votes, fps, w, h, original_frames
# ============================================================
# PROCESSAMENTO DO VÍDEO (2 PASSAGENS)
# ============================================================
def process_video(model, classifier, local_video, original_name):

    base_name = os.path.splitext(original_name)[0]

    stable_dir = "output"
    os.makedirs(stable_dir, exist_ok=True)
    output_video = os.path.join(stable_dir, f"{base_name}_annotated.mp4")

    frames_dir = os.path.join("output_frames", base_name)
    labels_dir = "output_labels"
    labels_path = os.path.join(labels_dir, f"{base_name}.txt")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # ---------- PASSAGEM 1 ----------
    frame_dets, votes, fps, w, h, original_frames = analyse_video(
        model, classifier, local_video, base_name
    )

    # ---------- escolher as 2 equipas ----------
    kits = []
    kit_scores = None

    if classifier is not None and votes is not None and votes.acc:
        kits, kit_scores = choose_kits(votes, classifier.names)

        top = np.argsort(-kit_scores)[:4]
        resumo = ", ".join(
            f"{classifier.names[int(i)]}={kit_scores[int(i)]:.1f}" for i in top
        )
        print(f"[teams] tracks com votos: {len(votes.acc)} | top classes: {resumo}")
        if len(kits) > 0:
            print(f"[teams] Team A = {classifier.names[kits[0]]}")
        if len(kits) > 1:
            print(f"[teams] Team B = {classifier.names[kits[1]]}")
    else:
        print("[teams] sem votos do classifier: tudo fica como Player")

    # ---------- LOG COMPLETO ----------
    log = {
        "video_id": base_name,
        "kits": [classifier.names[k] for k in kits] if kits else [],
        "kit_scores": {},
        "tracks": {}
    }

    if kit_scores is not None:
        log["kit_scores"] = {
            classifier.names[i]: float(kit_scores[i])
            for i in range(len(kit_scores))
        }

    # ---------- PASSAGEM 2: desenhar ----------
    cap = cv2.VideoCapture(local_video)
    labels_file = open(labels_path, "w")

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (w, h)
    )

    for frame_index, dets in enumerate(frame_dets):
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_index:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)

        # usar frame original (não anotado)
        original_frame = original_frames[frame_index]

        for e in dets:
            x1, y1, x2, y2 = e["box"]
            label = e["label"]
            tid = e["track_id"]

            # equipa final
            team = None

            if label == "player":
                if kits:
                    if tid is not None and votes is not None and tid in votes.acc:
                        team = team_from_votes(votes.acc[tid], kits)
                    elif e["frame_probs"] is not None:
                        team = team_from_votes(e["frame_probs"], kits)

                label_str = draw_entity(original_frame, "player", team, x1, y1, x2, y2)
            else:
                label_str = draw_entity(original_frame, label, None, x1, y1, x2, y2)

            labels_file.write(f"{frame_name} {label_str}\n")

            # LOG por track_id
            if tid not in log["tracks"]:
                log["tracks"][tid] = {
                    "label": label,
                    "frames": {},
                    "final_team": None,
                    "votes": votes.acc.get(tid).tolist() if tid in votes.acc else None,
                    "obs": votes.obs.get(tid) if tid in votes.obs else 0
                }

            log["tracks"][tid]["frames"][frame_index] = {
                "box": e["box"],
                "team": team,
                "probs": e["frame_probs"].tolist() if e["frame_probs"] is not None else None
            }

        # guardar frame original
        cv2.imwrite(frame_path, original_frame)
        writer.write(original_frame)

    cap.release()
    writer.release()
    labels_file.close()

    # equipa final por track_id
    for tid, tdata in log["tracks"].items():
        if tdata["votes"] is not None:
            tdata["final_team"] = team_from_votes(np.array(tdata["votes"]), kits)

    # guardar LOG
    log_path = os.path.join("output", f"{base_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    upload_file(log_path, f"processed/logs/{base_name}.json")

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

        # download vídeo + JSON
        local_video, local_json, json_key = download_video_and_json(key)
        delete_incoming(key, json_key)

        with open(local_json, "r") as f:
            data = json.load(f)

        user_email = data.get("email")
        user_name = data.get("name")
        start_time = int(data.get("start", 0))
        duration = int(data.get("duration", 0))

        # cortar vídeo se necessário
        cut_path = cut_video(local_video, start_time, duration) if duration > 0 else local_video
        safe_video = convert_video_to_safe_format(cut_path)

        # processar vídeo (2 passagens)
        output_video, frames_dir, labels_path = process_video(
            model, classifier, safe_video, os.path.basename(key)
        )

        # uploads essenciais
        upload_file(output_video, f"processed/{os.path.basename(output_video)}")
        upload_file(local_video, f"processed/originals/{os.path.basename(key)}")

        # upload dos frames ORIGINAIS (não anotados)
        for root, dirs, files_local in os.walk(frames_dir):
            for f in files_local:
                upload_file(
                    os.path.join(root, f),
                    f"processed/frames/{os.path.basename(frames_dir)}/{f}"
                )

        # upload das labels
        upload_file(labels_path, f"processed/labels/{os.path.basename(labels_path)}")

        # URL final do vídeo anotado
        video_url = (
            f"https://f003.backblazeb2.com/file/"
            f"{os.getenv('VIDEOS_BUCKET', 'handball-videos')}/"
            f"processed/{os.path.basename(output_video)}"
        )

        # criar página WordPress
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

        # enviar email ao utilizador
        send_email(user_email, user_name, page_url)

        # limpeza local
        shutil.rmtree("output", ignore_errors=True)
        shutil.rmtree("output_frames", ignore_errors=True)
        shutil.rmtree("output_labels", ignore_errors=True)
        shutil.rmtree("debug_crops", ignore_errors=True)

        os.remove(local_video)
        os.remove(safe_video)
        os.remove(local_json)


if __name__ == "__main__":
    main()
