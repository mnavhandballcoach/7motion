import torch
import cv2
import numpy as np

class TeamClassifier:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)  # ou torch.load, depende do teu export
        self.model.eval()

    def preprocess(self, crop):
        # redimensionar para o input do teu modelo
        img = cv2.resize(crop, (224, 224))  # ajusta se o teu modelo usar outro tamanho
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = torch.tensor(img).unsqueeze(0)
        return img

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return None

        inp = self.preprocess(crop)
        with torch.no_grad():
            out = self.model(inp)

        # assume que o modelo devolve logits [teamA, teamB]
        pred = torch.argmax(out, dim=1).item()

        return "A" if pred == 0 else "B"


def load_team_classifier():
    model_path = "team_classifier.pt"  # mete aqui o caminho real
    return TeamClassifier(model_path)
