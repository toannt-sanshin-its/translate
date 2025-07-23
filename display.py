import faiss
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# --- 1) Load FAISS index và metadata ---
index = faiss.read_index("indexes/faiss.index")
with open("indexes/meta.pkl", "rb") as f:
    metas = pickle.load(f)

# --- 2) Sample ngẫu nhiên up to 500 vectors ---
N = index.ntotal
sample_size = min(500, N)
indices = np.random.choice(N, sample_size, replace=False)

# --- 3) Reconstruct sampled vectors ---
sample_vectors = np.vstack([index.reconstruct(int(i)) for i in indices])

# --- 4) Lấy nhãn group để gán màu (source_id trước '_chunk') ---
labels = [metas[int(i)]["id"].split("_chunk")[0] for i in indices]
unique_labels = sorted(set(labels))
label_to_color = {lab: idx for idx, lab in enumerate(unique_labels)}
colors = [label_to_color[l] for l in labels]

# --- 5) Dimensionality Reduction ---
# 5.1) PCA xuống 50D để speed up
pca50 = PCA(n_components=50)
vecs_50d = pca50.fit_transform(sample_vectors)

# 5.2) t-SNE xuống 2D cho scatter đẹp mắt
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    init="pca",
    random_state=42,
    verbose=1
)
proj2d = tsne.fit_transform(vecs_50d)

# --- 6) Vẽ scatter 2D với màu phân groups ---
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    proj2d[:, 0],
    proj2d[:, 1],
    c=colors,
    cmap="tab20",
    s=30,
    alpha=0.8
)
plt.title("t-SNE Projection of Sampled Embeddings (2D)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
cbar.ax.set_yticklabels(unique_labels)
cbar.set_label("Source Documents", rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

# --- 7) (Tuỳ chọn) Vẽ 3D PCA Scatter ---
proj3d = PCA(n_components=3).fit_transform(sample_vectors)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    proj3d[:, 0],
    proj3d[:, 1],
    proj3d[:, 2],
    c=colors,
    cmap="tab20",
    s=30,
    alpha=0.8
)
ax.set_title("PCA Projection of Sampled Embeddings (3D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()
