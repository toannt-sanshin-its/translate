#!/usr/bin/env python
"""
Tạo FAISS index từ file .jsonl có format:
{
  "id": "...",
  "text": "...",           # tiếng Nhật
  "metadata": {
      "language": "ja",
      "translation": "...",# tiếng Việt (nếu đã có)
      ...
  }
}

Chạy:
    python build_index.py data/ja_vi.jsonl indexes/
"""

import argparse
import json
import pathlib
import pickle
import os
import numpy as np
import hashlib
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. Đọc tham số CLI ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS index cho dữ liệu Nhật-Việt (có chunking)"
    )
    parser.add_argument("jsonl_path", help="Đường dẫn file .jsonl")
    parser.add_argument("out_dir", help="Thư mục ghi index/metadata")
    parser.add_argument(
        "--max_tokens", type=int, default=256,
        help="Số token tối đa mỗi chunk"
    )
    parser.add_argument(
        "--stride", type=int, default=50,
        help="Overlap token giữa các chunk"
    )
    return parser.parse_args()

# --- 2. Hash fingerprint để dedup text ---
def fingerprint(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# --- 3. Load data và chunking ---
def load_and_chunk(jsonl_path, splitter):
    texts, metas = [], []
    for obj in load_jsonl(jsonl_path):
        if obj.get("metadata", {}).get("language") != "ja":
            continue
        chunks = splitter.split_text(obj["text"])
        for idx, txt in enumerate(chunks):
            texts.append(txt)
            metas.append({
                "id": f"{obj['id']}_chunk{idx}",
                "text": txt,
                "translation": obj["metadata"].get("translation", ""),
            })
    return texts, metas

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    args = parse_args()

    # --- Thiết lập model & tokenizer ---
    print("🔄 Nạp model embedding ...")
    emb_model = SentenceTransformer(
        os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        use_fast=True
    ) # Chia chuỗi text thành token IDs rồi ngược lại từ IDs thành text.

    # Tạo Recursive splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # các separator ưu tiên cắt: ngắt câu Nhật, dấu xuống dòng đôi, rồi ký tự bất kỳ
        separators=["。", "！", "？", "\n\n", ""],
        chunk_size=args.max_tokens,
        chunk_overlap=args.stride,
        length_function=lambda x: len(hf_tokenizer.encode(x))
    )

    texts, metas = [], []
    print("📥 Đọc & chunk dữ liệu... sử dụng CharacterTextSplitter")
    for obj in load_jsonl(args.jsonl_path):
        if obj.get("metadata", {}).get("language") != "ja":
            continue
        chunks = text_splitter.split_text(obj["text"])
        for idx, txt in enumerate(chunks):
            texts.append(txt)
            metas.append({
                "id": f"{obj['id']}_chunk{idx}",
                "text": txt,
                # chỉ giữ translation để tiết kiệm metadata
                "translation": obj["metadata"].get("translation", ""),
            })

    # --- Encode vectors ---
    print("⚙️ Encoding vectors...")
    vecs = emb_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True          # L2‑norm → cosine = inner‑product
    ).astype("float32")

    # --- Build FAISS index ---
    dim = vecs.shape[1]    # chiều, đang dùng là 384 chiều
    print('vecs', vecs)
    print('dim', dim)
    index = faiss.IndexFlatIP(dim)        # IP = cosine do vec đã chuẩn hoá
    index.add(vecs)

    # --- Ghi ra đĩa ---
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "meta.pkl", "wb") as fp: # Mở file ở chế độ nhị phân (“wb”)
        pickle.dump(metas, fp) # biến thành 1 luồng dữ liệu nhị phân (gọi là pickle stream) và ghi vào file

    print(f"✅ Đã lập chỉ mục {len(texts)} chunks → {out_dir}")

if __name__ == "__main__":
    main()
