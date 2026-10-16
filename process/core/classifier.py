# core/classifier.py Claude

import numpy as np
from ultralytics import YOLO


class TeamClassifier:
    def __init__(self, model_path):
        print(f"[Classifier] Loading team classifier model: {model_path}")
        self.model = YOLO(model_path)
        self.names = self.model.names  # {idx: nome}
        self.n_classes = len(self.names)

    def predict_batch(self, crops):
        """
        crops: lista de imagens BGR (numpy).
        Devolve array (N, n_classes) com as probabilidades de cada crop.
        Uma única chamada ao modelo para todos os jogadores do frame.
        """
        if not crops:
            return np.empty((0, self.n_classes), dtype=np.float32)

        results = self.model(crops, verbose=False)
        return np.stack([r.probs.data.cpu().numpy() for r in results])

    def predict_with_confidence(self, crop):
        """
        Compatibilidade com o pipeline antigo: (classe, confiança) de um crop.
        """
        probs = self.predict_batch([crop])[0]
        idx = int(probs.argmax())
        return self.names[idx], float(probs[idx])

    def predict(self, crop):
        """Compatibilidade com o pipeline antigo: só a classe."""
        return self.predict_with_confidence(crop)[0]
