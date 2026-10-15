# core/cropper.py

import cv2
import numpy as np

MIN_CROP_SIZE = 150  # tamanho mínimo real para ver camisola inteira
PAD_RATIO = 0.60     # expandir 60% do maior lado

def crop_with_pad(frame, x1, y1, x2, y2, w, h):
    """
    Crop quadrado, grande, centrado no jogador.
    Nunca reduz o bounding box original — só expande.
    """

    w_box = x2 - x1
    h_box = y2 - y1
    box_size = max(w_box, h_box)

    # pad grande
    pad = int(box_size * PAD_RATIO)

    # centro do jogador
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # crop quadrado
    half = (box_size // 2) + pad

    x1p = max(0, cx - half)
    y1p = max(0, cy - half)
    x2p = min(w, cx + half)
    y2p = min(h, cy + half)

    crop = frame[y1p:y2p, x1p:x2p]

    if crop is None or crop.size == 0:
        return None, (x1p, y1p, x2p, y2p)

    ch, cw = crop.shape[:2]

    # garantir tamanho mínimo
    if ch < MIN_CROP_SIZE or cw < MIN_CROP_SIZE:
        return None, (x1p, y1p, x2p, y2p)

    return crop, (x1p, y1p, x2p, y2p)


def is_box_too_small(w_box, h_box):
    """
    Filtra bounding boxes demasiado pequenas.
    Agora muito mais exigente.
    """
    return w_box < 50 or h_box < 80
