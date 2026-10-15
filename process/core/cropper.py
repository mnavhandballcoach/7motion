# core/cropper.py

import cv2

# Tamanho mínimo aceitável para um crop ser enviado ao classificador
MIN_CROP_SIZE = 120  # aumentámos para garantir camisola visível

def compute_dynamic_pad(w_box, h_box):
    """
    Pad dinâmico para garantir que o crop inclui tronco + camisola.
    Nunca reduzimos o box original — só expandimos.
    """
    pad = int(max(w_box, h_box) * 0.40)  # antes 0.20 → agora 40%
    return max(pad, 50)  # mínimo 50 px


def crop_with_pad(frame, x1, y1, x2, y2, w, h):
    """
    Faz o crop com pad dinâmico.
    Nunca reduz o bounding box original.
    Se o crop final for demasiado pequeno, devolve None.
    """
    w_box = x2 - x1
    h_box = y2 - y1

    pad = compute_dynamic_pad(w_box, h_box)

    # expandimos o box original
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    crop = frame[y1p:y2p, x1p:x2p]

    if crop is None or crop.size == 0:
        return None, (x1p, y1p, x2p, y2p)

    ch, cw = crop.shape[:2]

    # crop demasiado pequeno → não classificar
    if ch < MIN_CROP_SIZE or cw < MIN_CROP_SIZE:
        return None, (x1p, y1p, x2p, y2p)

    return crop, (x1p, y1p, x2p, y2p)


def is_box_too_small(w_box, h_box):
    """
    Filtra bounding boxes demasiado pequenas.
    Agora mais exigente para evitar ruído.
    """
    return w_box < 40 or h_box < 70
