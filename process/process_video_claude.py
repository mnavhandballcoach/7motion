import os
import cv2
import json
import shutil
import numpy as np
import requests

from process.core.models import load_yolo_model, load_team_classifier
from process.core.cropper import crop_player, is_box_too_small
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

# Um jogador só recebe equipa se pelo menos esta fração dos seus votos
# estiver nas 2 classes de kit escolhidas para o vídeo (0 a 1).
MIN_TEAM_MASS = float(os.getenv("TEAM_MIN_MASS", "0.10"))

# Para escolher as 2 equipas do vídeo, só contam tracks com pelo menos
# este número de observações (evita tracks curtos e ruidosos).
MIN_TRACK_OBS = int(os.getenv("TEAM_MIN_TRACK_OBS", "5"))

# Classes de kit a ignorar na escolha das equipas (separadas por vírgula).
# Ex.: TEAM_EXCLUDE_CLASSES="preto_solido" se for o kit de árbitro/guarda-redes.
EXCLUDE_CLASSES = {
    c.strip().lower()
    for c in os.getenv("TEAM_EXCLUDE_CLASSES", "").split(",")
    if c.strip()
}

# Crops de debug: desligados por defeito. Ativar com DEBUG_CROPS=1.
DEBUG_CROPS = os.getenv("DEBUG_CROPS", "0") == "1"
DEBUG_EVERY = int(os.getenv("DEBUG_CROPS_EVERY", "30"))


# ============================================================
# VOTAÇÃO POR TRACK_ID
# ============================================================
class TrackVotes:
    """
    Soma as probabilidades do classifier ao longo do tempo para cada
    track_id. Crops maiores (jogador mais perto da câmara) pesam mais.
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.acc = {}   # track_id -> vetor de votos acumulados
        self.obs = {}   # track_id -> nº de observações

    def update(self, track_id, probs, weight):
        if track_id not in self.acc:
            self.acc[track_id] = np.zeros(self.n_classes, dtype=np.float64)
            self.obs[track_id] = 0
        self.acc[track_id] += probs * weight
        self.obs[track_id] += 1


def choose_kits(votes, names):
    """
    Escolhe as 2 classes de kit mais presentes no vídeo.
    Cada jogador (track) conta uma vez, repartido pelas suas classes.
    Devolve ([idx_A, idx_B], scores_por_classe).
    """
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
    """
    Decide "A" ou "B" olhando só para as 2 classes de kit escolhidas.
    Uma previsão errada noutra classe (ex.: preto_solido) não conta.
    Devolve None se quase nada dos votos cair nessas 2 classes.
    """
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

    if cls_name is not None and conf is not None:
        fname = f"track_{track_id}_{cls_name}_{conf:.2f}.jpg"
    else:
        fname = f"track_{track_id}.jpg"

    local_path = os.path.join(debug_dir, fname)
    cv2.imwrite(local_path, crop)

    if video_id is not None:
        upload_debug_crop(local_path, video_id, frame_index, fname)


# ============================================================
# PASSAGEM 1: DETEÇÃO + TRACKING + VOTAÇÃO (sem desenhar)
# ============================================================
def analyse_video(model, classifier, local_video, base_name):
    cap = cv2.VideoCapture(local_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    votes = TrackVotes(classifier.n_classes) if classifier is not None else None
    frame_dets = []   # uma lista de deteções por frame
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Nesta passagem nunca se desenha nada, logo `frame` é sempre limpo.
        # persist=False no 1.º frame reinicia o tracker (importante entre vídeos)
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
                if not is_box_too_small(w_box, h_box):
                    # box inteira, igual ao treino do classifier
                    entry["crop"] = crop_player(frame, x1, y1, x2, y2, w, h)
                players.append(entry)

        # classificação em lote + acumulação de votos por track_id
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

                    if DEBUG_CROPS and frame_index % DEBUG_EVERY == 0:
                        idx = int(pr.argmax())
                        save_debug_crop(
                            p["crop"], frame_index, p["track_id"],
                            classifier.names[idx], float(pr[idx]), base_name
                        )

        # libertar memória: os crops já não são precisos
        for e in dets:
            e["crop"] = None

        frame_dets.append(dets)
        frame_index += 1

    cap.release()
    return frame_dets, votes, fps, w, h


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

    # ---------- passagem 1 ----------
    frame_dets, votes, fps, w, h = analyse_video(
        model, classifier, local_video, base_name
    )

    # ---------- escolher as 2 equipas do vídeo ----------
    kits = []
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

    # ---------- passagem 2: desenhar ----------
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

        for e in dets:
            x1, y1, x2, y2 = e["box"]
            label = e["label"]

            if label == "player":
                team = None
                tid = e["track_id"]

                if kits:
                    if tid is not None and votes is not None and tid in votes.acc:
                        team = team_from_votes(votes.acc[tid], kits)
                    elif e["frame_probs"] is not None:
                        team = team_from_votes(e["frame_probs"], kits)

                label_str = draw_entity(frame, "player", team, x1, y1, x2, y2)
            else:
                # referee / ball / goalkeeper: sem equipa
                label_str = draw_entity(frame, label, None, x1, y1, x2, y2)

            labels_file.write(f"{frame_name} {label_str}\n")

        cv2.imwrite(frame_path, frame)
        writer.write(frame)

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
        shutil.rmtree("debug_crops", ignore_errors=True)

        os.remove(local_video)
        os.remove(safe_video)
        os.remove(local_json)


if __name__ == "__main__":
    main()
