import torch
import torchvision.transforms as transforms
from PIL import Image
from .utils import laplacian_variance

from ultralytics import YOLO

IMG_SIZE = 224

class TeamClassifier:
    def __init__(self, model_path):
        print("[TeamClassifier] Loading YOLO classification model...")

        # Carrega o modelo YOLO corretamente
        self.model = YOLO(model_path)

        # YOLO já trata device internamente
        self.device = self.model.device

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        self.classes = ["Team A", "Team B"]

    def preprocess(self, crop_bgr):
        img = Image.fromarray(crop_bgr[:, :, ::-1])
        return self.transform(img)

    def predict(self, crop_bgr):
        # YOLO classification aceita diretamente imagens numpy
        results = self.model(crop_bgr)

        # top1 da classificação YOLO
        pred = results[0].probs.top1

        raw = self.classes[pred]

        if raw == "Team A":
            return "A"
        if raw == "Team B":
            return "B"
        return None


def classify_team(crop, classifier):
    if crop is None or crop.size == 0:
        return None

    if laplacian_variance(crop) < 15:
        return None

    try:
        return classifier.predict(crop)
    except Exception as e:
        print("[TeamClassifier] Error:", e)
        return None
