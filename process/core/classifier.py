from ultralytics import YOLO

class TeamClassifier:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, crop):
        """
        Retorna apenas a classe (compatível com pipeline antigo).
        """
        results = self.model(crop, verbose=False)
        probs = results[0].probs
        cls_idx = int(probs.top1)
        return self.model.names[cls_idx]

    def predict_with_confidence(self, crop):
        """
        Retorna (classe, confiança) — necessário para fusão com KMeans.
        """
        results = self.model(crop, verbose=False)
        probs = results[0].probs

        cls_idx = int(probs.top1)
        conf = float(probs.top1conf)

        cls_name = self.model.names[cls_idx]

        return cls_name, conf
