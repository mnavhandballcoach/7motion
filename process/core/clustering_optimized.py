import numpy as np
from sklearn.cluster import KMeans
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

# ============================
# 1. MobileNetV3 Embeddings (10× mais rápido)
# ============================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = mobilenet_v3_large(weights="IMAGENET1K_V1").to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

def extract_embedding_fast(crop_bgr):
    img = Image.fromarray(crop_bgr[:, :, ::-1])
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)

    return emb.cpu().numpy().flatten()


# ============================
# 2. HSV fallback
# ============================

def hsv_feature(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].flatten()
    s = hsv[:, :, 1].flatten()
    hist_h = np.histogram(h, bins=32, range=(0, 180))[0]
    hist_s = np.histogram(s, bins=32, range=(0, 255))[0]
    return np.concatenate([hist_h, hist_s]).astype(np.float32)


# ============================
# 3. Recorte da camisola (melhorado)
# ============================

def crop_jersey_only(crop):
    h = crop.shape[0]
    top = int(h * 0.25)
    bottom = int(h * 0.65)
    return crop[top:bottom, :]


# ============================
# 4. Clustering inicial (k=5)
# ============================

def run_initial_clustering(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.cluster_centers_, kmeans.labels_


# ============================
# 5. Identificar clusters das equipas
# ============================

def identify_team_clusters(cluster_labels):
    counts = np.bincount(cluster_labels)
    team_clusters = counts.argsort()[-2:]  # dois maiores
    return team_clusters


# ============================
# 6. Decidir Team A / Team B
# ============================

def decide_team_A_B(team_clusters, bboxes, cluster_labels):
    left_positions = {c: [] for c in team_clusters}

    for (x1, _, _, _), cid in zip(bboxes, cluster_labels):
        if cid in team_clusters:
            left_positions[cid].append(x1)

    team_A = min(left_positions, key=lambda c: np.mean(left_positions[c]))
    team_B = [c for c in team_clusters if c != team_A][0]

    return team_A, team_B


# ============================
# 7. Atribuir cluster por centróide
# ============================

def assign_cluster(embedding, centroids):
    dists = np.linalg.norm(centroids - embedding, axis=1)
    return int(np.argmin(dists))


# ============================
# 8. Atribuir equipa
# ============================

def assign_team(cluster_id, team_A, team_B):
    if cluster_id == team_A:
        return "A"
    if cluster_id == team_B:
        return "B"
    return None
