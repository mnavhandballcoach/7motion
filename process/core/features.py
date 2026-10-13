import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vit_b_16

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vit_b_16(weights="IMAGENET1K_V1").to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_embedding(crop_bgr):
    img = Image.fromarray(crop_bgr[:, :, ::-1])
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)
    return emb.cpu().numpy().flatten()
