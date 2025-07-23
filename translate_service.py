"""
FastAPI dịch Nhật -> Việt.
Ưu tiên bản dịch đã lưu trong FAISS (nếu gần đúng),
ngược lại fallback sang NLLB.

Chạy:
    uvicorn translate_service:app --reload
"""

import pickle, faiss, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import config as C  # import cấu hình ở trên

"""
pickle: để load file metadata pickle chứa thông tin bản dịch.
faiss: thư viện Facebook AI Similarity Search, dùng để lưu - truy vấn index embedding.
numpy: xử lý mảng số học, convert vector để FAISS search.
FastAPI, HTTPException:
    FastAPI: framework web dùng để expose API.
    HTTPException: raise lỗi trả về cho client (ví dụ khi input rỗng).
pydantic.BaseModel: Dùng define schema cho request/response, tự validate và tự sinh OpenAPI.

SentenceTransformer: Model embedding (ví dụ all-MiniLM…) để chuyển câu Nhật thành vector.

AutoTokenizer, AutoModelForSeq2SeqLM: Từ HuggingFace Transformers, để load tokenizer + model NLLB-200.
torch: ánh xạ model PyTorch lên GPU/CPU.
import config as C: Tất cả tham số cấu hình (đường dẫn index, threshold, model names, thiết lập device…) gom chung trong config.py
"""

# ---------- Khởi tạo ----------
app = FastAPI(title="JP→VI Translator (FAISS + NLLB)")

print("🚀 Loading embedding model ...")
emb_model = SentenceTransformer(C.EMB_MODEL, device=C.DEVICE)

print("🚀 Loading FAISS index ...")
faiss_index = faiss.read_index(C.INDEX_PATH)
metas = pickle.load(open(C.META_PATH, "rb"))

print("🚀 Loading NLLB model ...")
tok = AutoTokenizer.from_pretrained(C.NLLB_MODEL)
nllb = AutoModelForSeq2SeqLM.from_pretrained(C.NLLB_MODEL).to(C.DEVICE)

# ---------- Request / Response schemas ----------
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translation: str
    source: str          # "database" hoặc "nllb"
    score: float | None  # cosine similarity nếu dùng DB

# ---------- Helper ----------
def translate_with_nllb(jp_text: str) -> str:
    """
    Dịch bằng NLLB‑200 (jpn_Jpan -> vie_Latn) :contentReference[oaicite:2]{index=2}
    """
    inputs = tok(
        jp_text,
        return_tensors="pt",
        src_lang=C.SRC_LANG
    ).to(C.DEVICE)

    output_tokens = nllb.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids(C.TGT_LANG),
        max_length=256
    )
    return tok.batch_decode(output_tokens, skip_special_tokens=True)[0]

# ---------- API ----------
@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")

    # 1) Embed & truy vấn FAISS
    vec = emb_model.encode(
        req.text,
        normalize_embeddings=True,
    ).astype("float32")
    D, I = faiss_index.search(np.expand_dims(vec, 0), C.TOP_K)  # D: cosine scores

    best_idx, best_score = int(I[0][0]), float(D[0][0])

    if best_score >= C.SIM_THRESHOLD:
        # Có bản dịch tin cậy trong DB
        meta = metas[best_idx]
        return TranslationResponse(
            translation=meta["metadata"]["translation"],
            source="database",
            score=best_score
        )

    # 2) Nếu không đủ tin cậy → NLLB
    vi = translate_with_nllb(req.text)
    return TranslationResponse(
        translation=vi,
        source="nllb",
        score=best_score
    )
