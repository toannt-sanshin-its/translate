#!/usr/bin/env python
"""
CLI dịch Nhật → Anh dùng RAG (FAISS) + LLaMA.cpp (GGUF)

• Tìm top-k (mặc định 3) câu liên quan từ VectorDB (FAISS)
• Ghép chúng làm ngữ cảnh + câu hỏi vào prompt
• Gọi model local (GGUF) qua llama-cpp-python để sinh bản dịch (nhưng máy local ko hỗ trợ nên dùng tạm ctransformers)

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
import argparse # parse CLI flags
import json
import os
import pickle  # load metadata index
import sys
from typing import List, Tuple
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM        #  wrapper gọi LLaMA.cpp model
from config import Config

# ===================== TIỆN ÍCH ===================== #
def load_resources(cfg: Config):
    """Tải tất cả tài nguyên một lần."""
    # Load Embedding model
    emb = SentenceTransformer(cfg.EMB_MODEL, device="cpu")  # encode CPU đủ nhanh cho câu ngắn

    # Load FAISS index & metadata
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
       model_type="llama",                 # Nâng cao: Thay LLaMA bằng Llama 2 7B/13B hoặc Mistral 7B/8x7B (GGUF) cho chất lượng cao hơn.
    #    model_type="mistral",                 # Nâng cao: Thay LLaMA bằng Llama 2 7B/13B hoặc Mistral 7B/8x7B (GGUF) cho chất lượng cao hơn.
       gpu_layers=0,                       # CPU CPU-only (gpu_layers=0) , Nếu có GPU, set gpu_layers=-1 hoặc tăng số layer GPU để chạy nhanh hơn.
       context_length=cfg.N_CTX,           # Nâng cao: Quản lý context dài hơn: tăng cfg.N_CTX (ví dụ up to 4096 hoặc 8192 nếu model hỗ trợ)
       threads=cfg.THREADS,
   )
    # llm = AutoModel.from_pretrained(
    #     cfg.MODEL_PATH,            # đường dẫn tới .gguf của bạn
    #     model_type="llama",        # Vicuna xây trên kiến trúc Llama
    #     backend="gguf",            # định dạng file
    #     n_ctx=cfg.N_CTX,           # context window
    #     threads=cfg.THREADS
    # )

    try:
        actual_path = getattr(llm, "model_path", None)
        print(f"✔️ Model path in LLM object: {actual_path}")
    except Exception:
        pass

    return emb, index, metas, llm

# embedding câu tiếng nhật thành vector float32
def embed_sentence(emb_model: SentenceTransformer, text: str, normalize: bool = True) -> np.ndarray:
    vec = emb_model.encode(text, normalize_embeddings=normalize).astype("float32")
    return vec

# truy vấn top‑k nearest neighbors (faiss inner product/L2).
def search_topk(index: faiss.Index, vec: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
    D, I = index.search(np.expand_dims(vec, 0), k)        # Nâng cao: Khám phá ANN index như HNSW (IndexHNSWFlat) cho tốc độ truy vấn nhanh hơn trên dữ liệu lớn
    scores = D[0].tolist()
    idxs = I[0].tolist()
    return scores, idxs

# Truy nested dict an toàn
def safe_get(meta: dict, *keys, default: str = "") -> str:
    """Lấy giá trị từ dict lồng nhau, nếu không có trả default."""
    cur = meta
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default

def build_context(metas, idxs, scores, jp, cfg: Config) -> str:
    seen_vi = set()
    chunks = []

    for idx, score in zip(idxs, scores):
        # 1. Bỏ qua nếu score < ngưỡng
        if score < cfg.MIN_SCORE:
            continue

        # 2. Lấy bản dịch mẫu (VI)
        m      = metas[idx]
        vi_ex  = safe_get(m, "translation") \
               or safe_get(m, "metadata", "translation")
        if not vi_ex:
            continue

        # 3. Lọc duplicate
        if vi_ex in seen_vi:
            continue
        seen_vi.add(vi_ex)

        # 4. Thêm vào chunks
        chunks.append(f"- Example: {vi_ex}")

        # # 5. Giới hạn số ví dụ
        # if len(chunks) >= getattr(cfg, "CONTEXT_EXAMPLES", 2):
        #     break

    # Nếu không có example nào, trả về chuỗi rỗng để LLM tự fallback
    return "\n".join(chunks)

def call_ctfm(llm, cfg: Config, context: str, query: str) -> str:
    # kết hợp system + user prompt thành 1 chuỗi
    user_prompt = cfg.USER_PROMPT_TEMPLATE.format(context=context, query=query)

    prompt = f"<<SYS>>{cfg.SYSTEM_PROMPT}<<SYS>>\n{user_prompt}\n"
    print('==================== Prompt ========================')
    print(prompt)
    print("====================================================")
    return llm(
        prompt,
        temperature=cfg.TEMPERATURE,
        max_new_tokens=cfg.MAX_TOKENS,      # limit độ dài output.
        stop=["</s>", "<<SYS>>"]   # tuỳ model, có thể bỏ
    ).strip()

def translate_one(emb, index, metas, llm, cfg: Config, jp: str) -> Tuple[str, str, float]:
    # 1. Embed & search
    vec = embed_sentence(emb, jp, normalize=cfg.NORMALIZE_EMB)
    scores, idxs = search_topk(index, vec, cfg.TOP_K)
    print("============================= Score =======================")
    print(scores)
    print("===========================================================")
    best_score = scores[0]

    # 2. Build context chỉ khi neighbor đầu tiên đủ tốt
    if best_score >= cfg.MIN_SCORE:
        context = build_context(metas, idxs, scores, jp, cfg)
    else:
        context = ""
    vi = call_ctfm(llm, cfg, context, jp)
    return vi, "llama_ctx", float(scores[0])

def get_device_label(cfg: Config) -> str:
    return "GPU" if getattr(cfg, "gpu_layers", 0) and cfg.gpu_layers > 0 else "CPU"

# ===================== MAIN ===================== #
def parse_args():
    p = argparse.ArgumentParser(description="CLI dịch Nhật → Anh (RAG + LLaMA)")
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
    device = get_device_label(cfg)

    # Một câu
    if args.text:
        start_time = time.perf_counter()
        vi, src, score = translate_one(emb, index, metas, llm, cfg, args.text)
        last_time = time.perf_counter() - start_time
        if args.json:
            print(json.dumps({"src": src, "score": score, "jp": args.text, "en": vi}, ensure_ascii=False))
        else:
            print(f"[{src} | {score:.4f}] {vi}")
        
        with open("translate_log.txt", "a", encoding="utf-8") as logf:
            logf.write(
                f"TIME: {last_time:.4f}\tSCORE: {score:.4f}\tDEVICE: {cfg.DEVICE}\tJP: {args.text}\tEN: {vi}\n"
            )
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
