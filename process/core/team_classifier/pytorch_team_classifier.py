import torch
import torchvision.transforms as transforms
from PIL import Image

IMG_SIZE = 224

class TeamClassifier:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TeamClassifier] Using device:", self.device)

        self.model = torch.load(model_path, map_location=self.device)
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

        return self.classes[pred]
