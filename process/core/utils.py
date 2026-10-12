import numpy as np
import cv2

def parse_team_classifier_label(label_str):
    label_str = label_str.strip().lower()
    if "team a" in label_str:
        return "A"
    if "team b" in label_str:
        return "B"
    return None

def laplacian_variance(image):
    if image is None or image.size == 0:
        return 0.0
    return cv2.Laplacian(image, cv2.CV_64F).var()
