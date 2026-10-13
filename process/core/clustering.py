import numpy as np
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vit_b_16

# ============================
# 1. DINOv2 Embeddings
# ============================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ViT-B/16 pré-treinado (robusto para padrões de camisola)
model = vit_b_16(weights="IMAGENET1K_V1").to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_embedding(crop_bgr):
    """
    Extrai embedding DINOv2/ViT-B16 de um crop BGR.
    """
    img = Image.fromarray(crop_bgr[:, :, ::-1])
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)

    return emb.cpu().numpy().flatten()


# ============================
# 2. Recorte só da camisola
# ============================

def crop_jersey_only(crop):
    """
    Recorta apenas o tronco (20% a 60% da altura).
    Remove pernas, fundo e chão.
    """
    h = crop.shape[0]
    top = int(h * 0.20)
    bottom = int(h * 0.60)
    return crop[top:bottom, :]


# ============================
# 3. Clustering inicial
# ============================

def run_initial_clustering(embeddings, n_clusters=4):
    """
    Corre KMeans nos primeiros frames.
    Retorna centróides e labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.cluster_centers_, kmeans.labels_


# ============================
# 4. Identificar clusters das equipas
# ============================

def identify_team_clusters(cluster_labels):
    """
    Identifica os dois clusters maiores (equipas).
    Os restantes são GR/árbitros.
    """
    counts = np.bincount(cluster_labels)
    team_clusters = counts.argsort()[-2:]  # dois maiores
    return team_clusters


# ============================
# 5. Decidir Team A e Team B
# ============================

def decide_team_A_B(team_clusters, bboxes, cluster_labels):
    """
    Regra universal:
    - equipa mais à esquerda no início → Team A
    - outra → Team B
    """
    left_positions = {c: [] for c in team_clusters}

    for (x1, _, _, _), cid in zip(bboxes, cluster_labels):
        if cid in team_clusters:
            left_positions[cid].append(x1)

    # cluster com menor média de x1 → Team A
    team_A = min(left_positions, key=lambda c: np.mean(left_positions[c]))
    team_B = [c for c in team_clusters if c != team_A][0]

    return team_A, team_B


# ============================
# 6. Atribuir cluster por centróide
# ============================

def assign_cluster(embedding, centroids):
    """
    Atribui o embedding ao centróide mais próximo.
    """
    dists = np.linalg.norm(centroids - embedding, axis=1)
    return int(np.argmin(dists))


# ============================
# 7. Atribuir equipa ao jogador
# ============================

def assign_team(cluster_id, team_A, team_B):
    """
    Converte cluster → equipa.
    """
    if cluster_id == team_A:
        return "A"
    if cluster_id == team_B:
        return "B"
    return None  # GR ou árbitro
