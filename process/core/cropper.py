# core/cropper.py

import cv2

# Tamanho mínimo aceitável para um crop ser enviado ao classificador
MIN_CROP_SIZE = 80  # podes ajustar para 70–100 se quiseres

def compute_dynamic_pad(w_box, h_box):
    """
    Calcula um pad dinâmico em função do tamanho da bounding box.
    Aumentámos o pad para apanhar melhor tronco + camisola.
    """
    pad = int(max(w_box, h_box) * 0.35)  # antes 0.20
    return max(pad, 40)  # antes 20


def crop_with_pad(frame, x1, y1, x2, y2, w, h):
    """
    Faz o crop com pad dinâmico e garante que o crop não é demasiado pequeno.
    Se o crop final for muito pequeno, devolve None para o crop.
    """
    w_box = x2 - x1
    h_box = y2 - y1

    pad = compute_dynamic_pad(w_box, h_box)

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    crop = frame[y1p:y2p, x1p:x2p]

    # Proteção extra: se o crop for demasiado pequeno, não vale a pena classificar
    if crop is None or crop.size == 0:
        return None, (x1p, y1p, x2p, y2p)

    ch, cw = crop.shape[:2]
    if ch < MIN_CROP_SIZE or cw < MIN_CROP_SIZE:
        # demasiado pequeno para o classificador de padrões de camisola
        return None, (x1p, y1p, x2p, y2p)

    return crop, (x1p, y1p, x2p, y2p)


def is_box_too_small(w_box, h_box):
    """
    Filtra bounding boxes demasiado pequenas.
    Aqui estamos um pouco mais exigentes para evitar ruído.
    """
    return w_box < 35 or h_box < 60  # antes 25 / 40
