#!/usr/bin/env python
"""
CLI dịch Nhật → Việt dùng RAG (FAISS) + LLaMA.cpp (GGUF)

• Tìm top-k (mặc định 3) câu liên quan từ VectorDB (FAISS)
• Ghép chúng làm ngữ cảnh + câu hỏi vào prompt
• Gọi model local (GGUF) qua llama-cpp-python để sinh bản dịch

Ví dụ:
    python translate_cli_llama_rag.py \
      --text "患者は末梢神経障害を伴う高血圧を呈した。"

Batch:
    python translate_cli_llama_rag.py \
      --input_file jp_sentences.txt \
      --output_file vi_trans.txt

Yêu cầu:
    pip install sentence-transformers faiss-cpu llama-cpp-python==0.2.90 transformers

Gợi ý build GPU cho llama-cpp-python:
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-binary :all:
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from config import Config

# ===================== TIỆN ÍCH ===================== #
def load_resources(cfg: Config):
    """Tải tất cả tài nguyên một lần."""
    # Embedding model
    emb = SentenceTransformer(cfg.EMB_MODEL, device="cpu")  # encode CPU đủ nhanh cho câu ngắn

    # FAISS index & metadata
    if not os.path.exists(cfg.INDEX_PATH):
        sys.exit(f"❌ Không tìm thấy INDEX_PATH: {cfg.INDEX_PATH}")
    index = faiss.read_index(cfg.INDEX_PATH)

    if not os.path.exists(cfg.META_PATH):
        sys.exit(f"❌ Không tìm thấy META_PATH: {cfg.META_PATH}")
    with open(cfg.META_PATH, "rb") as f:
        metas = pickle.load(f)

    # LLaMA model
    if not os.path.exists(cfg.MODEL_PATH):
        sys.exit(f"❌ Không tìm thấy MODEL_PATH: {cfg.MODEL_PATH}")

    llm = AutoModelForCausalLM.from_pretrained(
       "models",
       model_file=os.path.basename(cfg.MODEL_PATH),
       model_type="llama",
       gpu_layers=0,                       # CPU
       context_length=cfg.N_CTX,
       threads=cfg.THREADS,
   )
    return emb, index, metas, llm

# embedding câu tiếng nhật thành vector float32
def embed_sentence(emb_model: SentenceTransformer, text: str, normalize: bool = True) -> np.ndarray:
    vec = emb_model.encode(text, normalize_embeddings=normalize).astype("float32")
    return vec


def search_topk(index: faiss.Index, vec: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
    D, I = index.search(np.expand_dims(vec, 0), k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    return scores, idxs


def safe_get(meta: dict, *keys, default: str = "") -> str:
    """Lấy giá trị từ dict lồng nhau, nếu không có trả default."""
    cur = meta
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default


def build_context(metas, idxs: List[int]) -> str:
    chunks = []
    for i in idxs:
        m = metas[i]
        # tuỳ cấu trúc metadata của bạn. Ở đây thử nhiều khả năng
        jp = safe_get(m, "jp", default="") or safe_get(m, "metadata", "jp", default="")
        vi = safe_get(m, "translation", default="") or safe_get(m, "metadata", "translation", default="")
        # Nếu chỉ cần dịch, có thể giữ JP/VI hay chỉ JP. Ta để cả hai để model hiểu phong cách
        chunk = f"- JP: {jp}\n  VI: {vi}" if vi else f"- JP: {jp}"
        chunks.append(chunk)
    return "\n\n".join(chunks)


def call_ctfm(llm, cfg: Config, context: str, query: str) -> str:
    # kết hợp system + user prompt thành 1 chuỗi
    user_prompt = cfg.USER_PROMPT_TEMPLATE.format(top_k=cfg.TOP_K, context=context, query=query)
    prompt = f"<<SYS>>{cfg.SYSTEM_PROMPT}<<SYS>>\n{user_prompt}\n"
    print('==================== Prompt ========================')
    print(prompt)
    print("====================================================")
    return llm(
        prompt,
        temperature=cfg.TEMPERATURE,
        max_new_tokens=cfg.MAX_TOKENS,
        stop=["</s>", "<<SYS>>"]   # tuỳ model, có thể bỏ
    ).strip()

def translate_one(emb, index, metas, llm, cfg: Config, jp: str) -> Tuple[str, str, float]:
    vec = embed_sentence(emb, jp, normalize=cfg.NORMALIZE_EMB)
    scores, idxs = search_topk(index, vec, cfg.TOP_K)
    context = build_context(metas, idxs)
    print('==================== Context ========================')
    print(context)
    print("=====================================================")
    vi = call_ctfm(llm, cfg, context, jp)
    return vi, "llama_ctx", float(scores[0])

# ===================== MAIN ===================== #
def parse_args():
    p = argparse.ArgumentParser(description="CLI dịch Nhật → Việt (RAG + LLaMA)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Câu tiếng Nhật cần dịch")
    g.add_argument("--input_file", help="File .txt, mỗi dòng một câu JP")
    p.add_argument("--output_file", help="File ghi kết quả (nếu batch)")
    p.add_argument("--json", action="store_true", help="Xuất kết quả dạng JSON (stdout)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config()

    emb, index, metas, llm = load_resources(cfg)

    # Một câu
    if args.text:
        vi, src, score = translate_one(emb, index, metas, llm, cfg, args.text)
        if args.json:
            print(json.dumps({"src": src, "score": score, "jp": args.text, "vi": vi}, ensure_ascii=False))
        else:
            print(f"[{src} | {score:.4f}] {vi}")
        return

    # Batch
    with open(args.input_file, encoding="utf-8") as fin:
        lines = [l.strip() for l in fin if l.strip()]

    outputs = []
    json_out = []
    for jp in lines:
        vi, src, score = translate_one(emb, index, metas, llm, cfg, jp)
        if args.json:
            json_out.append({"src": src, "score": score, "jp": jp, "vi": vi})
        else:
            outputs.append(f"[{src} | {score:.4f}] {vi}")

    if args.output_file:
        if args.json:
            with open(args.output_file, "w", encoding="utf-8") as fout:
                json.dump(json_out, fout, ensure_ascii=False, indent=2)
            print(f"✅ Đã ghi JSON {len(json_out)} dòng vào {args.output_file}")
        else:
            with open(args.output_file, "w", encoding="utf-8") as fout:
                fout.write("\n".join(outputs))
            print(f"✅ Đã ghi {len(outputs)} dòng vào {args.output_file}")
    else:
        if args.json:
            print(json.dumps(json_out, ensure_ascii=False))
        else:
            for o in outputs:
                print(o)


if __name__ == "__main__":
    main()
