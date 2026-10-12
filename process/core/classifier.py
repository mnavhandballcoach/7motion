import torch
import torchvision.transforms as transforms
from PIL import Image
from .utils import laplacian_variance

# 🔥 PyTorch 2.6 fix — permitir a classe usada no checkpoint YOLO
from torch.serialization import add_safe_globals
import ultralytics.nn.tasks as ul_tasks
add_safe_globals([ul_tasks.ClassificationModel])

IMG_SIZE = 224

class TeamClassifier:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[TeamClassifier] Using device:", self.device)

        # Agora funciona no PyTorch 2.6
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        self.classes = ["Team A", "Team B"]

    def preprocess(self, crop_bgr):
        img = Image.fromarray(crop_bgr[:, :, ::-1])
        return self.transform(img)

    def predict(self, crop_bgr):
        tensor = self.preprocess(crop_bgr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1).cpu().item()

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
