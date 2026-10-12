from ultralytics import YOLO
import numpy as np

IMG_SIZE = 224

class TeamClassifier:
    def __init__(self, model_path="bestclassifier.pt"):
        print("[TeamClassifier] Loading YOLO classification model...")
        self.model = YOLO(model_path)
        self.classes = ["A", "B"]

    def predict_batch(self, crops_bgr):
        if len(crops_bgr) == 0:
            return []

        # YOLO aceita lista de numpy arrays diretamente
        results = self.model(crops_bgr)

        preds = []
        for r in results:
            p = r.probs.top1
            if p < 0 or p >= len(self.classes):
                preds.append("UNK")
            else:
                preds.append(self.classes[p])

        return preds
