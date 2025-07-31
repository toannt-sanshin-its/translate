#!/usr/bin/env python
"""
Chạy:
    Chạy lần đầu init vectordb: python build_index.py data/ja_vi.jsonl indexes/
    Chạy để thêm data vào vectordb python build_index.py --extend data/new_chunks.jsonl indexes/
"""

import argparse
import pathlib
import pickle
import os
import faiss
from helper import fingerprint, load_jsonl, should_add, make_embedding_tools

# --- CLI ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS index cho dữ liệu Nhật-Việt (có chunking)"
    )
    parser.add_argument("jsonl_path", help="Input JSONL file")
    parser.add_argument("out_dir", help="Thư mục ghi index/metadata")
    parser.add_argument("--max_tokens", type=int, default=256, help="Số token tối đa mỗi chunk")
    parser.add_argument("--stride", type=int, default=50, help="Overlap token giữa các chunk")
    parser.add_argument("--extend", action='store_true', 
                        help="Extend existing index (with dedup)") # Nếu có --extend → gán args.extend = True; nếu không có → False
    return parser.parse_args()

# --- Load data và chunking ---
def load_and_chunk(jsonl_path, splitter, seen_fps=None):
    texts, metas = [], []
    dedup = seen_fps is not None
    for obj in load_jsonl(jsonl_path):
        if obj.get("metadata", {}).get("language") != "ja":
            continue
        chunks = splitter.split_text(obj["text"])
        for idx, txt in enumerate(chunks):
            if dedup:
                if not should_add(txt, seen_fps): # nếu có trong fingerprinter thì ko cho add vào db
                    continue
            texts.append(txt)
            metas.append({
                "id": f"doc{obj['id']}_chunk{idx}",
                "text": txt,
                "translation": obj["metadata"].get("translation", ""),
                "type": obj["metadata"].get("type", ""),
            })
    return texts, metas

# --- 1. Build mới (no dedup) ---
def init_index(jsonl_path: str, index_dir: pathlib.Path, max_tokens: int, stride: int):
    # Paths
    idx_path = index_dir / 'faiss.index'
    meta_path = index_dir / 'meta.pkl'
    fps_path = index_dir / 'seen_fps.pkl'

    # --- Thiết lập model & tokenizer ---
    model_name = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb_model, hf_tokenizer, text_splitter = make_embedding_tools(
        model_name,
        max_tokens=max_tokens,
        stride=stride
    )

    # Read & chunk
    print("📥 Đọc & chunk dữ liệu... sử dụng RecursiveCharacterTextSplitter")
    texts, metas = load_and_chunk(jsonl_path, text_splitter, seen_fps=None)
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
    index = faiss.IndexFlatIP(dim)        # IP = cosine do vec đã chuẩn hoá
    index.add(vecs)

    # Prepare seen_fps (all new)
    seen_fps = {fingerprint(txt) for txt in texts}

    # Persist
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    pickle.dump(metas, meta_path.open('wb'))
    pickle.dump(seen_fps, fps_path.open('wb'))

    print(f"✅ Initialized index: {len(texts)} chunks added.")

# --- 2. Extend (with dedup) ---
def extend_index(jsonl_path: str, index_dir: pathlib.Path, max_tokens: int, stride: int):
    # Paths
    idx_path = index_dir / 'faiss.index'
    meta_path = index_dir / 'meta.pkl'
    fps_path = index_dir / 'seen_fps.pkl'

    # Load previous state
    index = faiss.read_index(str(idx_path))
    metas = pickle.load(meta_path.open('rb'))
    seen_fps = pickle.load(fps_path.open('rb'))

    # --- Thiết lập model & tokenizer ---
    model_name = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb_model, hf_tokenizer, text_splitter = make_embedding_tools(
        model_name,
        max_tokens=max_tokens,
        stride=stride
    )

    # Process new file
    print("📥 Đọc & chunk dữ liệu... sử dụng RecursiveCharacterTextSplitter")
    new_texts, new_metas = load_and_chunk(jsonl_path, text_splitter, seen_fps)
    # Batch encode & add
    new_count = len(new_texts)
    if new_count:
        vecs = emb_model.encode(new_texts, normalize_embeddings=True).astype('float32')
        index.add(vecs)
        metas.extend(new_metas)

    # Persist updated state
    faiss.write_index(index, str(idx_path))
    pickle.dump(metas, meta_path.open('wb'))
    pickle.dump(seen_fps, fps_path.open('wb'))

    print(f"✅ Extended index: {new_count} new chunks added."
          f" Total now: {index.ntotal}")

def main():
    args = parse_args()
    idx_dir = pathlib.Path(args.out_dir)

    if args.extend and idx_dir.joinpath('faiss.index').exists():
        extend_index(
            jsonl_path=args.jsonl_path,
            index_dir=idx_dir,
            max_tokens=args.max_tokens,
            stride=args.stride
        )
    else:
        init_index(
            jsonl_path=args.jsonl_path,
            index_dir=idx_dir,
            max_tokens=args.max_tokens,
            stride=args.stride
        )

# ----------------------- MAIN ------------------------        
if __name__ == "__main__":
    main()
