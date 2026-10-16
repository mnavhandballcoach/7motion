# core/cropper.py Claude

import cv2
import numpy as np

def is_player_far(x1, y1, x2, y2, frame_w, frame_h):
    w_box = x2 - x1
    h_box = y2 - y1
    area_box = w_box * h_box
    area_frame = frame_w * frame_h

    # critérios profissionais
    too_short = h_box < frame_h * 0.12
    too_narrow = w_box < frame_w * 0.05
    too_small_area = area_box < area_frame * 0.02

    return too_short or too_narrow or too_small_area


def is_player_near(x1, y1, x2, y2, frame_w, frame_h):
    return not is_player_far(x1, y1, x2, y2, frame_w, frame_h)


def crop_player(frame, x1, y1, x2, y2, frame_w, frame_h):
    """
    Crop da BOX INTEIRA do jogador, igual ao que o YOLO usou no treino do
    classifier. NÃO redimensiona nem deforma: o Ultralytics aplica o mesmo
    pré-processamento do treino (resize do lado menor + center crop).

    IMPORTANTE: passar sempre um frame LIMPO (sem retângulos/textos desenhados).
    """
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop.copy()


def crop_torso_advanced(frame, x1, y1, x2, y2, frame_w, frame_h, out=224):
    """
    Recorta só o tronco (camisola), SEM deformar:
    - vertical: de 15% a 55% da altura da box (cabeça fora, calções fora)
    - horizontal: remove 15% de cada lado
    - padding até ficar quadrado (não estica) e só depois redimensiona

    Usar apenas se o classifier for treinado com este mesmo crop.
    """
    w_box = x2 - x1
    h_box = y2 - y1

    ty1 = int(y1 + h_box * 0.15)
    ty2 = int(y1 + h_box * 0.55)
    tx1 = int(x1 + w_box * 0.15)
    tx2 = int(x2 - w_box * 0.15)

    ty1, ty2 = max(0, ty1), min(frame_h, ty2)
    tx1, tx2 = max(0, tx1), min(frame_w, tx2)

    if ty2 <= ty1 or tx2 <= tx1:
        return None

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None

    # padding para quadrado (sem deformar)
    ch, cw = crop.shape[:2]
    s = max(ch, cw)
    canvas = np.zeros((s, s, 3), dtype=crop.dtype)
    oy = (s - ch) // 2
    ox = (s - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop

    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_AREA)


def crop_with_pad(frame, x1, y1, x2, y2, w, h):
    """
    Ajusta o bounding box para dar mais peso ao tronco.
    - Sobe o bounding box 25%
    - Corta 15% da parte inferior (pernas/chão)
    - Mantém a largura original
    """

    w_box = x2 - x1
    h_box = y2 - y1

    shift_up = int(h_box * 0.25)
    cut_bottom = int(h_box * 0.15)

    y1p = max(0, y1 - shift_up)
    y2p = min(h, y2 - cut_bottom)

    x1p = max(0, x1)
    x2p = min(w, x2)

    crop = frame[y1p:y2p, x1p:x2p]

    if crop is None or crop.size == 0:
        return None, (x1p, y1p, x2p, y2p)

    return crop, (x1p, y1p, x2p, y2p)


def is_box_too_small(w_box, h_box):
    """
    Filtra bounding boxes demasiado pequenas.
    Mantém apenas jogadores visíveis.
    """
    return w_box < 25 or h_box < 40
