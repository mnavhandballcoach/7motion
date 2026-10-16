# core/cropper.py

import cv2
import numpy as np

def crop_torso_advanced(frame, x1, y1, x2, y2, frame_w, frame_h):
    """
    Recorta o tronco do jogador de forma avançada:
    - remove cabeça
    - remove pernas
    - centra o tronco
    - replica o estilo do dataset
    - protege contra YOLO boxes demasiado altos/largos
    """

    w_box = x2 - x1
    h_box = y2 - y1

    # centro vertical do jogador
    center_y = y1 + h_box * 0.50

    # proporções ideais do dataset (ajustadas para riscas)
    torso_top = int(center_y - h_box * 0.35)   # sobe 35%
    torso_bottom = int(center_y + h_box * 0.45) # desce 45%

    # proteger contra limites da imagem
    torso_top = max(0, torso_top)
    torso_bottom = min(frame_h, torso_bottom)

    # remover fundo lateral (10% de cada lado)
    side_pad = int(w_box * 0.10)
    torso_x1 = max(0, x1 + side_pad)
    torso_x2 = min(frame_w, x2 - side_pad)

    crop = frame[torso_top:torso_bottom, torso_x1:torso_x2]

    if crop is None or crop.size == 0:
        return None

    # normalizar para 224x224 (igual ao dataset)
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

    return crop


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
