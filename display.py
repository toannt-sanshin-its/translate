import faiss
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

# --- Helper để chuẩn hóa màu ---
def make_color_mapping(labels):
    unique = sorted(set(labels))
    # Dùng colormap chu kỳ nếu quá nhiều label
    cmap = plt.get_cmap("tab20")
    color_list = [cmap(i % cmap.N) for i in range(len(unique))]
    label_to_color = {lab: color_list[idx] for idx, lab in enumerate(unique)}
    colors = [label_to_color[l] for l in labels]
    return label_to_color, colors

# --- 1) Load FAISS index và metadata ---
index = faiss.read_index("indexes/faiss.index")
with open("indexes/meta.pkl", "rb") as f:
    metas = pickle.load(f)

# Nếu là list, biến thành dict để dùng chung logic dễ hơn
if isinstance(metas, list):
    metas = {idx: entry for idx, entry in enumerate(metas)}

N = index.ntotal
if N == 0:
    raise RuntimeError("FAISS index rỗng (ntotal == 0); không có vector để hiển thị.")

# --- 2) Sample ngẫu nhiên up to 500 vectors ---
sample_size = min(500, N)
if sample_size == 0:
    raise RuntimeError("Không có mẫu nào để lấy (sample_size == 0).")

indices = np.random.choice(N, sample_size, replace=False)

# --- 3) Reconstruct sampled vectors (có catch lỗi nếu có thể) ---
reconstructed = []
valid_indices = []
for i in indices:
    try:
        vec = index.reconstruct(int(i))
        reconstructed.append(vec)
        valid_indices.append(int(i))
    except Exception as e:
        # Nếu không reconstruct được thì bỏ qua
        print(f"Warning: không thể reconstruct index {i}: {e}")

if len(reconstructed) == 0:
    raise RuntimeError("Không có vector hợp lệ sau khi reconstruct.")

sample_vectors = np.vstack(reconstructed)
actual_sample_size = sample_vectors.shape[0]

# --- 4) Lấy nhãn group để gán màu (source_id trước '_chunk') ---
labels = []
for i in valid_indices:
    # hỗ trợ cả dict và list metadata
    if isinstance(metas, dict):
        meta = metas.get(int(i), {})
    elif isinstance(metas, list):
        if 0 <= int(i) < len(metas):
            meta = metas[int(i)]
        else:
            meta = {}
    else:
        meta = {}

    raw_id = ""
    if isinstance(meta, dict):
        raw_id = meta.get("id", "")

    if "_chunk" in raw_id:
        label = raw_id.split("_chunk")[0]
    elif raw_id:
        label = raw_id
    else:
        label = "unknown"
    labels.append(label)

# tạo ánh xạ màu và danh sách màu từ labels
label_to_color, colors = make_color_mapping(labels)

# phòng ngừa bất thường: nếu số màu không khớp số vector thì fallback toàn bộ xám
if len(colors) != actual_sample_size:
    colors = ["gray"] * actual_sample_size

if not labels:
    # chẳng hạn chỉ gán một nhãn chung để vẫn có colors
    labels = ["unknown"] * actual_sample_size
    label_to_color, colors = make_color_mapping(labels)

# --- 5) Dimensionality Reduction ---
# 5.1) PCA xuống <=50D để tăng tốc; đảm bảo không vượt quá số mẫu/số chiều
n_features = sample_vectors.shape[1]
pca_target = min(50, actual_sample_size, n_features)
if pca_target >= 2:
    pca50 = PCA(n_components=pca_target, random_state=42)
    vecs_50d = pca50.fit_transform(sample_vectors)
else:
    # Nếu chỉ 1 chiều khả dụng thì giữ nguyên
    vecs_50d = sample_vectors.copy()

# 5.2) t-SNE xuống 2D chỉ khi có >=2 mẫu
do_tsne = actual_sample_size >= 2
if do_tsne:
    # perplexity phải nhỏ hơn số mẫu và >=2; thường để <= (n_samples-1)/3
    max_perp = max(2, (actual_sample_size - 1) / 3)
    perplexity = min(30, max_perp - 1e-6)  # đảm bảo strict < n_samples
    perplexity = max(2, perplexity)  # ít nhất 2
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        max_iter=1000,
        init="pca" if vecs_50d.shape[1] >= 2 else "random",
        random_state=42,
        verbose=1,
    )
    proj2d = tsne.fit_transform(vecs_50d)
else:
    proj2d = None  # không có để vẽ

# --- 6) Vẽ PCA 3D hoặc fallback ---
if actual_sample_size >= 3:
    n_comp_3d = min(3, actual_sample_size, n_features)
    pca3 = PCA(n_components=n_comp_3d, random_state=42)
    proj3d = pca3.fit_transform(sample_vectors)

    if proj3d.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            proj3d[:, 0],
            proj3d[:, 1],
            proj3d[:, 2],
            c=colors,
            s=40,
            alpha=0.8,
            edgecolors="w",
            linewidths=0.3,
        )
        ax.set_title("PCA Projection of Sampled Embeddings (3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.tight_layout()
        plt.show()
    else:
        # Nếu không đủ thành 3 chiều sau PCA (hiếm khi xảy ra), vẽ 2D
        plt.figure(figsize=(8, 6))
        plt.scatter(
            proj3d[:, 0],
            proj3d[:, 1],
            c=colors,
            s=40,
            alpha=0.8,
            edgecolors="w",
            linewidths=0.3,
        )
        plt.title("PCA Projection (fallback 2D, không đủ chiều cho 3D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()
elif actual_sample_size == 2:
    # Chỉ 2 điểm, làm PCA 2D
    pca2 = PCA(n_components=2, random_state=42)
    proj2d_pca = pca2.fit_transform(sample_vectors)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        proj2d_pca[:, 0],
        proj2d_pca[:, 1],
        c=colors,
        s=60,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.5,
    )
    plt.title("PCA Projection (2 điểm, 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
else:
    # Chỉ một vector, tự hiển thị điểm đơn 3D giả lập
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], c=colors, s=100, alpha=0.9, edgecolors="k")
    ax.set_title("Chỉ một embedding, không đủ dữ liệu cho PCA 3D")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    plt.show()
