#!/usr/bin/env python
"""
CLI dịch Nhật → Việt (FAISS + NLLB)

▶ Dịch một câu:
    python translate_cli.py \
      --text "患者は末梢神経障害を伴う高血圧を呈した。"

▶ Dịch batch nhiều câu (mỗi dòng 1 câu):
    python translate_cli.py \
      --input_file jp_sentences.txt \
      --output_file vi_trans.txt
"""
from __future__ import annotations
import argparse, pickle, faiss, numpy as np, torch, time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import config as C            # đọc biến môi trường đã định nghĩa

# ---------- Khởi tạo tài nguyên duy nhất ----------
def load_resources():
    # 1) Embedding model & FAISS
    emb = SentenceTransformer(C.EMB_MODEL, device=C.DEVICE)
    index = faiss.read_index(C.INDEX_PATH)
    metas = pickle.load(open(C.META_PATH, "rb"))
    # 2) NLLB
    tok = AutoTokenizer.from_pretrained(C.NLLB_MODEL)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(C.NLLB_MODEL).to(C.DEVICE)
    return emb, index, metas, tok, nllb

# ---------- Hàm tiện ích ----------
def translate_nllb(nllb, tok, jp: str) -> str:
    # 1) Prepend mã ngôn ngữ nguồn vào text
    text = f"{C.SRC_LANG} {jp}"        # C.SRC_LANG = "jpn_Jpan"
    # 2) Tokenize bình thường
    inputs = tok(text, return_tensors="pt").to(C.DEVICE)
    # 3) Forced BOS cho ngôn ngữ đích
    forced_bos = tok.convert_tokens_to_ids(C.TGT_LANG)  # C.TGT_LANG = "vie_Latn"
    # 4) Generate với beam search
    out_ids = nllb.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_length=256,
        num_beams=4,          # tinh chỉnh beam_search, nghĩa là bạn giữ song song 4 nhánh (“beam width = 4”)., tăng chất lượng nhưng chi phí tăng do phải đánh giá nhiều giả thuyết song song
        early_stopping=True   # tinh chỉnh beam_search
    )

    # 5) Decode, đồng thời bỏ token đích ở đầu nếu có
    result = tok.batch_decode(out_ids, skip_special_tokens=True)[0]
    # NLLB sẽ prefix bằng "<vie_Latn>" → chúng ta strip nó đi:
    return result.replace(f"{C.TGT_LANG} ", "")

def translate_one(emb, index, metas, tok, nllb, jp: str, log_file: str=None) -> tuple[str, str, float]:
    """Trả về (dịch_vi, nguồn, score_cosine)."""
    start = time.perf_counter()

    # thực hiện retrieval trong pineline
    vec = emb.encode(jp, normalize_embeddings=True).astype("float32") # chuyển câu jp thành vector embedding kính thước d (384 chìu)
                                                                      # FAISS chỉ chấp nhận vector float32 và tiết kiệm bộ nhớ so với float64.
                                                                      # vec có shape (d,) và dtype float32.
    D, I = index.search(np.expand_dims(vec, 0), C.TOP_K)    # D (distances/scores): shape (1, k), chứa giá trị inner-product (cosine) cao nhất với truy vấn, sắp xếp giảm dần.
                                                            # I (indices): shape (1, k), chứa chỉ số (0-based) của những vector tương ứng trong index.
    score, idx = float(D[0][0]), int(I[0][0])
    if score >= C.SIM_THRESHOLD:
        meta = metas[idx]
        if isinstance(meta, dict):
            # ưu tiên trường 'translation' ở top-level
            trans = meta.get("translation") or (meta.get("metadata") or {}).get("translation")
            if trans:
                trans_source = "database"
                elapsed = time.perf_counter() - start
    else:
        # Fallback to NLLB
        trans = translate_nllb(nllb, tok, jp)
        trans_source = "nllb"
        elapsed = time.perf_counter() - start

    # if score < C.SIM_THRESHOLD and log_file:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"TIME: {elapsed:.3f}s\tSCORE: {score:.4f}\tSRC: {trans_source}\t\tDEVICE: {C.DEVICE}\tJP: {jp}\tVI: {trans}\n")

    return trans, trans_source, score, elapsed

# ---------- Entry ----------
def main():
    parser = argparse.ArgumentParser(description="CLI dịch Nhật → Việt (RAG)")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Câu tiếng Nhật cần dịch")
    g.add_argument("--input_file", help="File .txt, mỗi dòng một câu JP")
    parser.add_argument("--output_file", help="File ghi kết quả (nếu batch)")
    parser.add_argument("--log", default="log_file.txt", help="File ghi các câu fallback (score thấp)")
    args = parser.parse_args()

    emb, index, metas, tok, nllb = load_resources()

    # --- 1 câu ---
    if args.text:
        vi, src, score, elapsed = translate_one(emb, index, metas, tok, nllb, args.text, args.log)
        print(f"[{src} | {score:.4f} | {elapsed:.3f}s] {vi}")
        return

    # --- Batch ---
    with open(args.input_file, encoding="utf-8") as fin:
        lines = [l.strip() for l in fin if l.strip()]

    outputs = []
    for jp in lines:
        vi, src, score = translate_one(emb, index, metas, tok, nllb, jp, args.log)
        outputs.append(f"[{src} | {score:.4f}] {vi}")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as fout:
            fout.write("\n".join(outputs))
        print(f"✅ Đã ghi {len(outputs)} dòng vào {args.output_file}")
    else:
        for o in outputs:
            print(o)

if __name__ == "__main__":
    main()
