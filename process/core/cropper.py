# core/cropper.py

import cv2
import numpy as np

def crop_with_pad(frame, x1, y1, x2, y2, w, h):
    """
    Ajusta o bounding box para dar mais peso ao tronco.
    - Sobe o bounding box 25%
    - Corta 15% da parte inferior (pernas/chão)
    - Mantém a largura original
    """

    w_box = x2 - x1
    h_box = y2 - y1

    # Subir o bounding box (para apanhar tronco + padrão)
    shift_up = int(h_box * 0.25)

    # Cortar parte inferior (pernas + chão)
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
