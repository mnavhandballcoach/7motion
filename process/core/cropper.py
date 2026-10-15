# core/cropper.py

import cv2
import numpy as np

def crop_with_pad(frame, x1, y1, x2, y2, w, h):
    """
    Crop EXACTO do bounding box da pessoa.
    Sem pad, sem expansão, sem ajustes.
    Igual ao treino.
    """

    # garantir limites
    x1p = max(0, x1)
    y1p = max(0, y1)
    x2p = min(w, x2)
    y2p = min(h, y2)

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
