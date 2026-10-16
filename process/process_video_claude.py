import os
import time
import cv2
import json
import shutil
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

from process.core.models import load_yolo_model, load_team_classifier
from process.core.cropper import crop_player
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

# Como decidir as equipas:
#   "color"      -> cor da camisola de cada jogador, agrupada em 2 equipas
#                   com KMeans (não precisa do classifier; funciona com
#                   equipas novas)
#   "classifier" -> modelo treinado (votação por track + 2 kits mais votados)
TEAM_MODE = os.getenv("TEAM_MODE", "color").strip().lower()

# ---- modo "color" ----
HUE_BINS = 12
# Tons de pele e de chão de madeira (laranja/amarelo) são ignorados.
# Pôr SKIN_HUE_LO=0 e SKIN_HUE_HI=0 desliga (se houver camisolas laranja).
SKIN_HUE_LO = int(os.getenv("SKIN_HUE_LO", "4"))
SKIN_HUE_HI = int(os.getenv("SKIN_HUE_HI", "24"))
# Tracks com pelo menos isto de observações servem para formar as 2 equipas.
COLOR_FIT_MIN_OBS = int(os.getenv("COLOR_FIT_MIN_OBS", "10"))
# Tracks com menos observações que isto ficam como Player.
COLOR_ASSIGN_MIN_OBS = int(os.getenv("COLOR_ASSIGN_MIN_OBS", "3"))
# Se a distância à equipa errada não for pelo menos X vezes a distância à
# certa, o jogador fica como Player (ambíguo).
COLOR_MIN_MARGIN = float(os.getenv("COLOR_MIN_MARGIN", "1.15"))
# Abaixo desta distância entre as 2 equipas o log avisa que as cores são
# muito parecidas.
COLOR_MIN_SEPARATION = float(os.getenv("COLOR_MIN_SEPARATION", "0.30"))

# ---- modo "classifier" ----
# Um jogador só recebe equipa se pelo menos esta fração dos seus votos
# estiver nas 2 classes de kit escolhidas para o vídeo (0 a 1).
MIN_TEAM_MASS = float(os.getenv("TEAM_MIN_MASS", "0.10"))
MIN_TRACK_OBS = int(os.getenv("TEAM_MIN_TRACK_OBS", "5"))
EXCLUDE_CLASSES = {
    c.strip().lower()
    for c in os.getenv("TEAM_EXCLUDE_CLASSES", "").split(",")
    if c.strip()
}
CLASSIFY_EVERY = int(os.getenv("CLASSIFY_EVERY", "3"))
WARMUP_OBS = int(os.getenv("CLASSIFY_WARMUP_OBS", "8"))

# ---- comum ----
# Tamanho mínimo da box para tentar identificar a equipa.
MIN_BOX_W = int(os.getenv("MIN_BOX_W", "16"))
MIN_BOX_H = int(os.getenv("MIN_BOX_H", "32"))

# Peso de cada observação = altura da box / (WEIGHT_REF * altura do frame),
# limitado entre WEIGHT_MIN e 1.
WEIGHT_REF = float(os.getenv("VOTE_WEIGHT_REF", "0.20"))
WEIGHT_MIN = float(os.getenv("VOTE_WEIGHT_MIN", "0.10"))

# Uploads em paralelo (resultado igual, só mais rápido).
UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "8"))

# Enviar para o Backblaze todos os frames anotados (processed/frames/...).
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
# FUNÇÕES COMUNS
# ============================================================
def box_too_small(w_box, h_box):
    return w_box < MIN_BOX_W or h_box < MIN_BOX_H


def vote_weight(box_h, frame_h):
    return float(min(1.0, max(WEIGHT_MIN, box_h / (WEIGHT_REF * float(frame_h)))))


def save_debug_crop(crop, frame_index, track_id, cls_name=None, conf=None):
    """Guarda o crop localmente; o envio é feito no fim, em paralelo."""
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
# MODO "color": COR DA CAMISOLA
# ============================================================
def torso_color_feature(frame, x1, y1, x2, y2, fw, fh):
    """
    Descritor da cor da camisola: zona central do tronco da box.
    Vetor = raiz de [histograma de tons (12 bins, só píxeis coloridos e sem
    pele/chão), fração de branco, fração de escuro].
    Devolve None se a zona for demasiado pequena.
    """
    w_box = x2 - x1
    h_box = y2 - y1

    tx1 = max(0, int(x1 + w_box * 0.25))
    tx2 = min(fw, int(x2 - w_box * 0.25))
    ty1 = max(0, int(y1 + h_box * 0.18))
    ty2 = min(fh, int(y1 + h_box * 0.52))

    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None

    roi = frame[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hch = hsv[..., 0]
    sch = hsv[..., 1]
    vch = hsv[..., 2]
    n = float(hch.size)

    dark = vch < 60
    white = (sch < 40) & (vch >= 150)
    colored = (sch >= 40) & (vch >= 60)

    if SKIN_HUE_HI > SKIN_HUE_LO:
        skin_or_floor = (hch >= SKIN_HUE_LO) & (hch <= SKIN_HUE_HI)
        colored = colored & ~skin_or_floor

    bins = (hch[colored].astype(np.int32) * HUE_BINS) // 180
    hist = np.bincount(bins, minlength=HUE_BINS)[:HUE_BINS].astype(np.float64) / n

    feat = np.concatenate([hist, [white.sum() / n, dark.sum() / n]])
    return np.sqrt(feat).astype(np.float32)


class TrackColors:
    """Média ponderada do descritor de cor de cada track."""

    def __init__(self):
        self.acc = {}   # track_id -> soma(feat * peso)
        self.w = {}     # track_id -> soma dos pesos
        self.n = {}     # track_id -> nº de observações

    def update(self, track_id, feat, weight):
        if track_id not in self.acc:
            self.acc[track_id] = np.zeros_like(feat, dtype=np.float64)
            self.w[track_id] = 0.0
            self.n[track_id] = 0
        self.acc[track_id] += feat.astype(np.float64) * weight
        self.w[track_id] += weight
        self.n[track_id] += 1

    def mean(self, track_id):
        return (self.acc[track_id] / max(self.w[track_id], 1e-9)).astype(np.float32)


def classify_color(feat, centroids):
    """
    Equipa mais próxima (distância euclidiana ao centro de cada equipa).
    Devolve (equipa ou None se ambíguo, dist_A, dist_B, margem).
    """
    da = float(np.linalg.norm(feat - centroids["A"]))
    db = float(np.linalg.norm(feat - centroids["B"]))

    near, d_near, d_far = ("A", da, db) if da <= db else ("B", db, da)
    ratio = d_far / max(d_near, 1e-6)

    team = near if ratio >= COLOR_MIN_MARGIN else None
    return team, da, db, ratio


def build_color_teams(colors):
    """
    Forma as 2 equipas com KMeans sobre os tracks com observações suficientes
    e atribui cada track à equipa mais próxima.
    Devolve (centroids|None, team_of_track, details, info).
    """
    info = {}

    fit = [t for t in colors.acc if colors.n[t] >= COLOR_FIT_MIN_OBS]
    info["fit_tracks"] = len(fit)
    if len(fit) < 2:
        return None, {}, {}, info

    feats = np.stack([colors.mean(t) for t in fit]).astype(np.float32)

    cv2.setRNGSeed(0)   # resultado repetível
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    _, labels, centers = cv2.kmeans(
        feats, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()

    sizes = [int((labels == k).sum()) for k in (0, 1)]
    if min(sizes) == 0:
        return None, {}, {}, info

    if sizes[0] != sizes[1]:
        a = 0 if sizes[0] > sizes[1] else 1
    else:
        obs = [
            sum(colors.n[t] for t, lab in zip(fit, labels) if lab == k)
            for k in (0, 1)
        ]
        a = 0 if obs[0] >= obs[1] else 1

    centroids = {"A": centers[a].copy(), "B": centers[1 - a].copy()}
    info["separation"] = float(np.linalg.norm(centers[0] - centers[1]))
    info["sizes"] = {"A": sizes[a], "B": sizes[1 - a]}

    team_of = {}
    details = {}
    for t in colors.acc:
        if colors.n[t] < COLOR_ASSIGN_MIN_OBS:
            team_of[t] = None
            details[t] = None
            continue
        team, da, db, ratio = classify_color(colors.mean(t), centroids)
        team_of[t] = team
        details[t] = (da, db, ratio)

    return centroids, team_of, details, info


def describe_centroid(c):
    """Texto legível de um centro de equipa: tom dominante, branco, escuro."""
    frac = np.square(c.astype(np.float64))
    hue_part = frac[:HUE_BINS]
    best = int(hue_part.argmax())
    hue_center = (best + 0.5) * 180.0 / HUE_BINS
    return (
        f"tom dominante H~{hue_center:.0f} ({hue_part[best]:.2f} da area) | "
        f"branco={frac[HUE_BINS]:.2f} | escuro={frac[HUE_BINS + 1]:.2f}"
    )


def log_color_teams(run_log, info, centroids):
    if centroids is None:
        run_log.add(
            f"[teams] cor: tracks suficientes para formar equipas = "
            f"{info.get('fit_tracks', 0)} (preciso de 2+) -> tudo fica como Player"
        )
        return

    sep = info["separation"]
    run_log.add(
        f"[teams] cor: tracks usados para formar equipas={info['fit_tracks']} | "
        f"A={info['sizes']['A']} tracks, B={info['sizes']['B']} tracks | "
        f"separacao={sep:.2f}"
    )
    if sep < COLOR_MIN_SEPARATION:
        run_log.add(
            f"[teams] AVISO: separacao abaixo de {COLOR_MIN_SEPARATION} "
            f"(cores das equipas muito parecidas, resultado pode ser pouco fiavel)"
        )
    run_log.add(f"[teams] Team A: {describe_centroid(centroids['A'])}")
    run_log.add(f"[teams] Team B: {describe_centroid(centroids['B'])}")


def log_tracks_color(run_log, track_info, colors, team_of, details):
    run_log.add("")
    run_log.add("---- tracks de jogadores (por ordem de aparição) ----")

    for tid, info in sorted(track_info.items(), key=lambda kv: kv[1]["first"]):
        n = info["n"]
        avg_h = info["sum_h"] / max(n, 1)
        head = (
            f"[track {tid}] frames {info['first']:05d}-{info['last']:05d} "
            f"(n={n}) alt_media={avg_h:.0f}px pequena={100 * info['small'] / max(n, 1):.0f}%"
        )

        if tid not in colors.acc:
            run_log.add(f"{head} | sem cor medida -> Player")
            continue

        obs = colors.n[tid]
        det = details.get(tid)
        if det is None:
            run_log.add(f"{head} obs_cor={obs} -> Player (poucas observacoes)")
            continue

        da, db, ratio = det
        team = team_of.get(tid)
        result = f"Team {team}" if team else "Player (ambiguo)"
        run_log.add(
            f"{head} obs_cor={obs} distA={da:.2f} distB={db:.2f} "
            f"margem={ratio:.1f}x -> {result}"
        )


# ============================================================
# MODO "classifier": VOTAÇÃO POR TRACK_ID
# ============================================================
class TrackVotes:
    """Soma as probabilidades do classifier ao longo do tempo por track_id."""

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.acc = {}
        self.obs = {}

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


def log_tracks_classifier(run_log, track_info, votes, kits, classifier):
    run_log.add("")
    run_log.add("---- tracks de jogadores (por ordem de aparição) ----")

    two = len(kits) == 2

    for tid, info in sorted(track_info.items(), key=lambda kv: kv[1]["first"]):
        n = info["n"]
        avg_h = info["sum_h"] / max(n, 1)
        head = (
            f"[track {tid}] frames {info['first']:05d}-{info['last']:05d} "
            f"(n={n}) alt_media={avg_h:.0f}px pequena={100 * info['small'] / max(n, 1):.0f}%"
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
# PASSAGEM 1: DETEÇÃO + TRACKING + MEDIÇÃO (sem desenhar)
# ============================================================
def analyse_video(model, classifier, local_video):
    cap = cv2.VideoCapture(local_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    use_color = TEAM_MODE == "color"
    use_classifier = (TEAM_MODE == "classifier") and classifier is not None

    votes = TrackVotes(classifier.n_classes) if use_classifier else None
    colors = TrackColors()

    frame_dets = []      # uma lista de deteções por frame
    track_info = {}      # track_id -> dados de todos os frames de jogador
    stats = {
        "detect_s": 0.0, "color_s": 0.0, "crop_s": 0.0, "classify_s": 0.0,
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
                "color": None,
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
                    {"first": frame_index, "last": frame_index,
                     "n": 0, "sum_h": 0.0, "small": 0},
                )
                info["last"] = frame_index
                info["n"] += 1
                info["sum_h"] += h_box

            if box_too_small(w_box, h_box):
                stats["players_small"] += 1
                if track_id is not None:
                    track_info[track_id]["small"] += 1
                continue

            if use_color:
                t1 = time.perf_counter()
                feat = torso_color_feature(frame, x1, y1, x2, y2, w, h)
                stats["color_s"] += time.perf_counter() - t1

                if feat is not None:
                    if track_id is not None:
                        colors.update(track_id, feat, vote_weight(h_box, h))
                    else:
                        entry["color"] = feat

                if DEBUG_CROPS and frame_index % DEBUG_EVERY == 0:
                    save_debug_crop(
                        crop_player(frame, x1, y1, x2, y2, w, h),
                        frame_index, track_id,
                    )

            elif use_classifier and should_classify(votes, track_id, frame_index):
                t1 = time.perf_counter()
                entry["crop"] = crop_player(frame, x1, y1, x2, y2, w, h)
                stats["crop_s"] += time.perf_counter() - t1

        # modo classifier: classificação em lote + votos por track_id
        if use_classifier:
            batch = [p for p in players if p["crop"] is not None]
            if batch:
                t2 = time.perf_counter()
                probs = classifier.predict_batch([p["crop"] for p in batch])
                stats["classify_s"] += time.perf_counter() - t2
                stats["crops"] += len(batch)

                for p, pr in zip(batch, probs):
                    if p["track_id"] is not None:
                        votes.update(p["track_id"], pr, vote_weight(p["box_h"], h))
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
    return frame_dets, votes, colors, track_info, stats, fps, w, h


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
        f"[config] TEAM_MODE={TEAM_MODE} MIN_BOX={MIN_BOX_W}x{MIN_BOX_H} "
        f"WEIGHT_REF={WEIGHT_REF} WEIGHT_MIN={WEIGHT_MIN} "
        f"SKIN_HUE={SKIN_HUE_LO}-{SKIN_HUE_HI} COLOR_FIT_MIN_OBS={COLOR_FIT_MIN_OBS} "
        f"COLOR_ASSIGN_MIN_OBS={COLOR_ASSIGN_MIN_OBS} COLOR_MIN_MARGIN={COLOR_MIN_MARGIN} "
        f"UPLOAD_WORKERS={UPLOAD_WORKERS} UPLOAD_FRAMES={int(UPLOAD_FRAMES)} "
        f"DEBUG_CROPS={int(DEBUG_CROPS)}"
    )

    # ---------- passagem 1 ----------
    t_p1 = time.perf_counter()
    frame_dets, votes, colors, track_info, stats, fps, w, h = analyse_video(
        model, classifier, local_video
    )
    p1_s = time.perf_counter() - t_p1
    n_frames = len(frame_dets)

    run_log.add(f"[video] {w}x{h} fps={fps:.2f} frames={n_frames}")
    run_log.add(
        f"[time] passagem 1 total: {p1_s:.1f}s "
        f"({1000 * p1_s / max(n_frames, 1):.0f} ms/frame) | "
        f"deteção+tracking: {stats['detect_s']:.1f}s | "
        f"cor: {stats['color_s']:.1f}s | crops: {stats['crop_s']:.1f}s | "
        f"classifier: {stats['classify_s']:.1f}s ({stats['crops']} crops)"
    )
    run_log.add(
        f"[players] deteções={stats['players']} | sem ID={stats['players_no_id']} | "
        f"pequenas demais={stats['players_small']} | "
        f"tracks distintos={len(track_info)}"
    )

    # ---------- decidir as equipas ----------
    centroids = None
    team_of = {}
    kits = []

    if TEAM_MODE == "color":
        centroids, team_of, details, info = build_color_teams(colors)
        log_color_teams(run_log, info, centroids)
        log_tracks_color(run_log, track_info, colors, team_of, details)

    else:
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

        log_tracks_classifier(run_log, track_info, votes, kits, classifier)

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

                if TEAM_MODE == "color":
                    if tid is not None and tid in team_of:
                        team = team_of[tid]
                    elif e["color"] is not None and centroids is not None:
                        team = classify_color(e["color"], centroids)[0]
                else:
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

    # O classifier só é preciso no modo "classifier".
    classifier = None
    if TEAM_MODE == "classifier":
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
