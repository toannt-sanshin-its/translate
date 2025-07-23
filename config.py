"""
Đọc biến môi trường & cung cấp hằng số cấu hình.
"""

import os
from dotenv import load_dotenv

# Nạp file .env ở thư mục gốc
load_dotenv()

# ---------- Embedding ----------
EMB_MODEL: str = os.getenv("EMB_MODEL")                         # ex: paraphrase-multilingual-MiniLM-L12-v2 -> có 384 chiều

# ---------- NLLB ----------
NLLB_MODEL: str = os.getenv("NLLB_MODEL")                      # ex: facebook/nllb-200-distilled-600M
SRC_LANG: str = "jpn_Jpan"                                     # mã FLoRes 200 cho tiếng Nhật :contentReference[oaicite:0]{index=0}
TGT_LANG: str = "vie_Latn"                                     # mã FLoRes 200 cho tiếng Việt :contentReference[oaicite:1]{index=1}

# ---------- FAISS ----------
INDEX_PATH: str = os.getenv("INDEX_PATH", "./indexes/faiss.index")
META_PATH: str  = os.getenv("META_PATH",  "./indexes/meta.pkl")

# ---------- Runtime ----------
DEVICE: str = os.getenv("DEVICE", "cpu")                       # "cuda" nếu có GPU
TOP_K: int = int(os.getenv("TOP_K", 3))                        # số câu gần nhất cần truy xuất
SIM_THRESHOLD: float = float(os.getenv("SIM_THRESHOLD", 0.85)) # ngưỡng cosine để tin cậy bản dịch sẵn
