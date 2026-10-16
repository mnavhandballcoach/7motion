import os
import time
import cv2
import json
import shutil
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

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
EXCLUDE_CLASSES = {
    c.strip().lower()
    for c in os.getenv("TEAM_EXCLUDE_CLASSES", "").split(",")
    if c.strip()
}

# Velocidade: cada jogador é classificado em todos os frames nas primeiras
# WARMUP observações e depois só 1 frame em cada CLASSIFY_EVERY.
# Frames seguidos são quase iguais, por isso quase não muda o resultado.
# CLASSIFY_EVERY=1 => classifica sempre (comportamento anterior).
CLASSIFY_EVERY = int(os.getenv("CLASSIFY_EVERY", "3"))
WARMUP_OBS = int(os.getenv("CLASSIFY_WARMUP_OBS", "8"))

# Uploads em paralelo (resultado igual, só mais rápido).
UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "8"))

# Enviar para o Backblaze todos os frames anotados (processed/frames/...).
# Pôr UPLOAD_FRAMES=0 se não precisares deles.
UPLOAD_FRAMES = os.getenv("UPLOAD_FRAMES", "1") == "1"

# Crops de debug: desligados por defeito. Ativar com DEBUG_CROPS=1.
DEBUG_CROPS = os.getenv("DEBUG_CROPS", "0") == "1"
DEBUG_EVERY = int(os.getenv("DEBUG_CROPS_EVERY", "30"))


# ============================================================
# LOG (imprime no terminal e guarda para enviar para processed/logs)
# ============================================================
class RunLog:
    def __init__(self, base_name):
        self.base_name = base_name
        self.lines = []

    def add(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self):
        os.makedirs("output_logs", exist_ok=True)
        path = os.path.join("output_logs", f"{self.base_name}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        return path


# ============================================================
# UPLOADS EM PARALELO (com 2.ª tentativa em série)
# ============================================================
def upload_many(pairs, run_log=None, what="ficheiros"):
    """pairs: lista de (caminho_local, caminho_remoto). Devolve nº de falhas."""
    if not pairs:
        return 0

    t0 = time.perf_counter()
    failed = []

    with ThreadPoolExecutor(max_workers=max(1, UPLOAD_WORKERS)) as ex:
        futures = [(pair, ex.submit(upload_file, pair[0], pair[1])) for pair in pairs]
        for pair, fut in futures:
            try:
                fut.result()
            except Exception as e:
                failed.append((pair, repr(e)))

    still_failed = 0
    for pair, err in failed:
        try:
            upload_file(pair[0], pair[1])
        except Exception as e:
            still_failed += 1
            if run_log:
                run_log.add(f"[upload] FALHOU {pair[1]}: {e!r}")

    if run_log:
        run_log.add(
            f"[time] upload de {len(pairs)} {what}: "
            f"{time.perf_counter() - t0:.1f}s (falhas: {still_failed})"
        )
    return still_failed


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
        self.obs = {}   # track_id -> nº de observações classificadas

    def update(self, track_id, probs, weight):
        if track_id not in self.acc:
            self.acc[track_id] = np.zeros(self.n_classes, dtype=np.float64)
            self.obs[track_id] = 0
        self.acc[track_id] += probs * weight
        self.obs[track_id] += 1


def should_classify(votes, track_id, frame_index):
    if votes is None or CLASSIFY_EVERY <= 1 or track_id is None:
        return True
    if votes.obs.get(track_id, 0) < WARMUP_OBS:
        return True
    return (frame_index + track_id) % CLASSIFY_EVERY == 0


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
# GUARDAR CROP DE DEBUG (o envio é feito no fim, em paralelo)
# ============================================================
def save_debug_crop(crop, frame_index, track_id, cls_name=None, conf=None):
    if crop is None:
        return

    debug_dir = os.path.join("debug_crops", f"frame_{frame_index:05d}")
    os.makedirs(debug_dir, exist_ok=True)

    if cls_name is not None and conf is not None:
        fname = f"track_{track_id}_{cls_name}_{conf:.2f}.jpg"
    else:
        fname = f"track_{track_id}.jpg"

    cv2.imwrite(os.path.join(debug_dir, fname), crop)


# ============================================================
# PASSAGEM 1: DETEÇÃO + TRACKING + VOTAÇÃO (sem desenhar)
# ============================================================
def analyse_video(model, classifier, local_video):
    cap = cv2.VideoCapture(local_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    votes = TrackVotes(classifier.n_classes) if classifier is not None else None
    frame_dets = []      # uma lista de deteções por frame
    track_info = {}      # track_id -> dados de todos os frames de jogador
    stats = {
        "detect_s": 0.0, "crop_s": 0.0, "classify_s": 0.0,
        "crops": 0, "players": 0, "players_no_id": 0, "players_small": 0,
    }
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Nesta passagem nunca se desenha nada, logo `frame` é sempre limpo.
        # persist=False no 1.º frame reinicia o tracker (importante entre vídeos)
        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=frame_index > 0,
            tracker=TRACKER,
            verbose=False,
        )
        stats["detect_s"] += time.perf_counter() - t0

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

            if base_label != "player":
                continue

            players.append(entry)
            stats["players"] += 1

            if track_id is None:
                stats["players_no_id"] += 1
            else:
                info = track_info.setdefault(
                    track_id,
                    {"first": frame_index, "last": frame_index, "n": 0, "sum_h": 0.0},
                )
                info["last"] = frame_index
                info["n"] += 1
                info["sum_h"] += h_box

            if is_box_too_small(w_box, h_box):
                stats["players_small"] += 1
                continue

            # só faz o crop se este jogador vai ser classificado neste frame
            if classifier is not None and should_classify(votes, track_id, frame_index):
                t1 = time.perf_counter()
                # box inteira, igual ao treino do classifier
                entry["crop"] = crop_player(frame, x1, y1, x2, y2, w, h)
                stats["crop_s"] += time.perf_counter() - t1

        # classificação em lote + acumulação de votos por track_id
        if classifier is not None:
            batch = [p for p in players if p["crop"] is not None]
            if batch:
                t2 = time.perf_counter()
                probs = classifier.predict_batch([p["crop"] for p in batch])
                stats["classify_s"] += time.perf_counter() - t2
                stats["crops"] += len(batch)

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
                            classifier.names[idx], float(pr[idx])
                        )

        # libertar memória: os crops já não são precisos
        for e in dets:
            e["crop"] = None

        frame_dets.append(dets)
        frame_index += 1

    cap.release()
    return frame_dets, votes, track_info, stats, fps, w, h


# ============================================================
# LOG POR JOGADOR (track)
# ============================================================
def log_tracks(run_log, track_info, votes, kits, classifier):
    run_log.add("")
    run_log.add("---- tracks de jogadores (por ordem de aparição) ----")

    two = len(kits) == 2

    for tid, info in sorted(track_info.items(), key=lambda kv: kv[1]["first"]):
        n = info["n"]
        avg_h = info["sum_h"] / max(n, 1)
        head = (
            f"[track {tid}] frames {info['first']:05d}-{info['last']:05d} "
            f"(n={n}) alt_media={avg_h:.0f}px"
        )

        vec = votes.acc.get(tid) if votes is not None else None
        if vec is None:
            run_log.add(f"{head} | sem crops/votos -> Player")
            continue

        total = float(vec.sum())
        obs = votes.obs.get(tid, 0)
        top = classifier.names[int(vec.argmax())]

        if two and total > 0:
            a = vec[kits[0]] / total
            b = vec[kits[1]] / total
            team = team_from_votes(vec, kits)
            result = f"Team {team}" if team else "Player (votos fora de A/B)"
            run_log.add(
                f"{head} obs={obs} A={a:.2f} B={b:.2f} top={top} -> {result}"
            )
        else:
            run_log.add(f"{head} obs={obs} top={top}")


# ============================================================
# PROCESSAMENTO DO VÍDEO (2 PASSAGENS)
# ============================================================
def process_video(model, classifier, local_video, original_name, run_log):

    base_name = os.path.splitext(original_name)[0]

    stable_dir = "output"
    os.makedirs(stable_dir, exist_ok=True)
    output_video = os.path.join(stable_dir, f"{base_name}_annotated.mp4")

    frames_dir = os.path.join("output_frames", base_name)
    labels_dir = "output_labels"
    labels_path = os.path.join(labels_dir, f"{base_name}.txt")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    run_log.add(f"=== LOG {base_name} | {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    run_log.add(
        f"[config] CLASSIFY_EVERY={CLASSIFY_EVERY} WARMUP_OBS={WARMUP_OBS} "
        f"TEAM_MIN_MASS={MIN_TEAM_MASS} TEAM_MIN_TRACK_OBS={MIN_TRACK_OBS} "
        f"EXCLUDE={sorted(EXCLUDE_CLASSES) or '-'} UPLOAD_WORKERS={UPLOAD_WORKERS} "
        f"UPLOAD_FRAMES={int(UPLOAD_FRAMES)} DEBUG_CROPS={int(DEBUG_CROPS)}"
    )

    # ---------- passagem 1 ----------
    t_p1 = time.perf_counter()
    frame_dets, votes, track_info, stats, fps, w, h = analyse_video(
        model, classifier, local_video
    )
    p1_s = time.perf_counter() - t_p1
    n_frames = len(frame_dets)

    run_log.add(f"[video] {w}x{h} fps={fps:.2f} frames={n_frames}")
    run_log.add(
        f"[time] passagem 1 total: {p1_s:.1f}s "
        f"({1000 * p1_s / max(n_frames, 1):.0f} ms/frame) | "
        f"deteção+tracking: {stats['detect_s']:.1f}s | "
        f"crops: {stats['crop_s']:.1f}s | "
        f"classifier: {stats['classify_s']:.1f}s ({stats['crops']} crops)"
    )
    run_log.add(
        f"[players] deteções={stats['players']} | sem ID={stats['players_no_id']} | "
        f"pequenas demais (sem crop)={stats['players_small']} | "
        f"tracks distintos={len(track_info)}"
    )

    # ---------- escolher as 2 equipas do vídeo ----------
    kits = []
    if classifier is not None and votes is not None and votes.acc:
        kits, kit_scores = choose_kits(votes, classifier.names)

        top = np.argsort(-kit_scores)[:5]
        resumo = ", ".join(
            f"{classifier.names[int(i)]}={kit_scores[int(i)]:.1f}" for i in top
        )
        run_log.add(f"[teams] tracks com votos: {len(votes.acc)} | top classes: {resumo}")
        if len(kits) > 0:
            run_log.add(f"[teams] Team A = {classifier.names[kits[0]]}")
        if len(kits) > 1:
            run_log.add(f"[teams] Team B = {classifier.names[kits[1]]}")
    else:
        run_log.add("[teams] sem votos do classifier: tudo fica como Player")

    log_tracks(run_log, track_info, votes, kits, classifier)

    # ---------- passagem 2: desenhar ----------
    t_p2 = time.perf_counter()

    cap = cv2.VideoCapture(local_video)
    labels_file = open(labels_path, "w")

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (w, h)
    )

    drawn = {"A": 0, "B": 0, "Player": 0}

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

                drawn[team if team else "Player"] += 1
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

    p2_s = time.perf_counter() - t_p2
    run_log.add("")
    run_log.add(f"[time] passagem 2 (desenhar e gravar): {p2_s:.1f}s")
    run_log.add(
        f"[resumo] deteções de jogador desenhadas: "
        f"Team A={drawn['A']} | Team B={drawn['B']} | Player={drawn['Player']}"
    )

    # ---------- enviar crops de debug (em paralelo) ----------
    if DEBUG_CROPS and os.path.isdir("debug_crops"):
        pairs = []
        for root, _, files_local in os.walk("debug_crops"):
            for f in files_local:
                local = os.path.join(root, f)
                rel = os.path.relpath(local, "debug_crops").replace(os.sep, "/")
                pairs.append((local, f"processed/crops/{base_name}/{rel}"))
        upload_many(pairs, run_log, "crops de debug")

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
        t_total = time.perf_counter()

        key = file["Key"]
        base_name = os.path.splitext(os.path.basename(key))[0]
        run_log = RunLog(base_name)

        local_video, local_json, json_key = download_video_and_json(key)
        delete_incoming(key, json_key)

        with open(local_json, "r") as f:
            data = json.load(f)

        user_email = data.get("email")
        user_name = data.get("name")
        start_time = int(data.get("start", 0))
        duration = int(data.get("duration", 0))

        t_prep = time.perf_counter()
        cut_path = cut_video(local_video, start_time, duration) if duration > 0 else local_video
        safe_video = convert_video_to_safe_format(cut_path)
        prep_s = time.perf_counter() - t_prep

        output_video, frames_dir, labels_path = process_video(
            model, classifier, safe_video, os.path.basename(key), run_log
        )
        run_log.add(f"[time] cortar/converter vídeo (ffmpeg): {prep_s:.1f}s")

        # ---------- uploads ----------
        upload_file(output_video, f"processed/{os.path.basename(output_video)}")
        upload_file(local_video, f"processed/originals/{os.path.basename(key)}")

        if UPLOAD_FRAMES:
            pairs = []
            for root, dirs, files_local in os.walk(frames_dir):
                for f in files_local:
                    pairs.append((
                        os.path.join(root, f),
                        f"processed/frames/{os.path.basename(frames_dir)}/{f}",
                    ))
            upload_many(pairs, run_log, "frames")

        upload_file(labels_path, f"processed/labels/{os.path.basename(labels_path)}")

        # ---------- guardar o log em processed/logs/<nome do vídeo>.log ----------
        run_log.add(f"[time] TOTAL até aqui: {time.perf_counter() - t_total:.1f}s")
        log_path = run_log.save()
        upload_file(log_path, f"processed/logs/{os.path.basename(log_path)}")

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
            "video_id": base_name,
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
        shutil.rmtree("output_logs", ignore_errors=True)
        shutil.rmtree("debug_crops", ignore_errors=True)

        os.remove(local_video)
        os.remove(safe_video)
        os.remove(local_json)


if __name__ == "__main__":
    main()
