import os
import requests
from ultralytics import YOLO

MODEL_URL = "https://github.com/mnavhandballcoach/7motion/releases/download/models/best.pt"
CLASSIFIER_URL = "https://github.com/mnavhandballcoach/7motion/releases/download/models/bestclassifier.pt"

def download_file(url, local_path):
    print(f"Downloading model from {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    print(f"Saved model to {local_path}")
    return local_path

def load_yolo_model():
    local_path = "best.pt"
    if not os.path.exists(local_path):
        download_file(MODEL_URL, local_path)
    return YOLO(local_path)

def load_team_classifier():
    """
    ⚠️ Este método agora só garante que o modelo PyTorch é descarregado.
    Não devolve YOLO, porque o classificador PyTorch é carregado em core/classifier.py.
    """
    local_path = "bestclassifier.pt"
    if not os.path.exists(local_path):
        download_file(CLASSIFIER_URL, local_path)
    return local_path
