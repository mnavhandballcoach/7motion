from .utils import parse_team_classifier_label, laplacian_variance

def adaptive_conf_min(w_box, h_box, w, h):
    area_ratio = (w_box * h_box) / (w * h + 1e-6)
    conf_min = 0.70 - area_ratio * 0.25
    return max(0.45, conf_min)

def classify_team(crop, classifier, w_box, h_box, w, h):
    if crop is None or crop.size == 0:
        return None

    if laplacian_variance(crop) < 15:
        return None

    cls_results = classifier(crop)
    team_idx = int(cls_results[0].probs.top1)
    conf = float(cls_results[0].probs.top1conf)
    raw_label = classifier.names[team_idx]

    conf_min = adaptive_conf_min(w_box, h_box, w, h)
    if conf >= conf_min:
        return parse_team_classifier_label(raw_label)
    return None
