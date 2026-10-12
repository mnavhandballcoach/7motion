def compute_dynamic_pad(w_box, h_box):
    pad = int(max(w_box, h_box) * 0.40)
    return max(pad, 35)

def crop_with_pad(frame, x1, y1, x2, y2, w, h):
    w_box = x2 - x1
    h_box = y2 - y1

    pad = compute_dynamic_pad(w_box, h_box)

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    crop = frame[y1p:y2p, x1p:x2p]
    return crop, (x1p, y1p, x2p, y2p)

def is_box_too_small(w_box, h_box):
    return w_box < 40 or h_box < 60
