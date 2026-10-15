from ultralytics import YOLO

class TeamClassifier:
    def __init__(self, model_path):
        print(f"[Classifier] Loading team classifier model: {model_path}")
        self.model = YOLO(model_path)

    def predict(self, crop):
        """
        Retorna apenas a classe (compatível com pipeline antigo),
        mas agora com LOGS para confirmar o comportamento.
        """
        results = self.model(crop, verbose=False)
        probs = results[0].probs

        cls_idx = int(probs.top1)
        cls_name = self.model.names[cls_idx]
        conf = float(probs.top1conf)

        print(f"[Classifier] Predicted class: {cls_name} (conf={conf:.2f})")

        return cls_name

    def predict_with_confidence(self, crop):
        """
        Retorna (classe, confiança) — necessário para fusão com KMeans.
        Também com logs.
        """
        results = self.model(crop, verbose=False)
        probs = results[0].probs

        cls_idx = int(probs.top1)
        conf = float(probs.top1conf)
        cls_name = self.model.names[cls_idx]

        print(f"[Classifier] Predicted class: {cls_name} (conf={conf:.2f})")

        return cls_name, conf
