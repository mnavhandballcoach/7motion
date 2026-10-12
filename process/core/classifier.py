from ultralytics import YOLO
import numpy as np
import cv2

IMG_SIZE = 224

class TeamClassifier:
    def __init__(self, model_path="bestclassifier.pt"):
        print("[TeamClassifier] Loading YOLO classification model...")
        self.model = YOLO(model_path)
        # IMPORTANT: pipeline trabalha com "A" / "B"
        self.classes = ["A", "B"]

    def predict_batch(self, crops_bgr):
        if len(crops_bgr) == 0:
            return []

        # Resize crops para o tamanho de treino
        crops_resized = []
        for c in crops_bgr:
            if c is None or c.size == 0:
                continue
            cr = cv2.resize(c, (IMG_SIZE, IMG_SIZE))
            crops_resized.append(cr)

        if len(crops_resized) == 0:
            return []

        results = self.model(crops_resized)

        preds = []
        for r in results:
            p = r.probs.top1
            if p < 0 or p >= len(self.classes):
                preds.append("UNK")
            else:
                preds.append(self.classes[p])

        return preds
