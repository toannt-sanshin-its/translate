# helper.py
import hashlib
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def fingerprint(text: str) -> str:
    """
    Tính MD5 fingerprint dùng để deduplication.
    Input:
      - text: chuỗi cần fingerprint
    Trả về:
      - fp: chuỗi hex 32 ký tự (MD5)
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def should_add(text: str, seen_fps: set) -> bool:
    """
    Kiểm tra xem `text` đã từng add chưa dựa trên fingerprint:

    - Nếu chưa: tính fingerprint, add vào `seen_fps`, trả về True (được phép add).
    - Nếu đã có: trả về False (skip).
    """
    fp = fingerprint(text)
    if fp in seen_fps:
        return False
    seen_fps.add(fp)
    return True

def load_jsonl(path):
    """
    Generator đọc file .jsonl, mỗi dòng trả về 1 dict.
    Usage:
      for obj in load_jsonl("data.jsonl"):
          ...
    """
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def ensure_dir(path):
    """
    Tạo thư mục nếu chưa tồn tại.
    Input:
      - path: pathlib.Path hoặc string
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_embedding_tools(model_name: str, max_tokens: int, stride: int):
    """
    Trả về tuple (emb_model, tokenizer, splitter) đã được cấu hình sẵn.

    - model_name: tên hoặc đường dẫn của embedding model
    - max_tokens: kích thước tối đa mỗi chunk
    - stride: overlap giữa các chunk
    """
    print("🔄 Nạp model embedding ...")
    # 1. Nạp model embedding
    emb_model = SentenceTransformer(model_name)

    # 2. Nạp tokenizer (dùng để tính độ dài token cho splitter)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) # Chia chuỗi text thành token IDs rồi ngược lại từ IDs thành text.

    # 3. Tạo RecursiveCharacterTextSplitter cho văn bản Nhật
    text_splitter = RecursiveCharacterTextSplitter(
        # các separator ưu tiên cắt: ngắt câu Nhật, dấu xuống dòng đôi, rồi ký tự bất kỳ
        separators=["。", "！", "？", "\n\n", ""],
        chunk_size=max_tokens,
        chunk_overlap=stride,
        length_function=lambda txt: len(hf_tokenizer.encode(txt))
    )

    return emb_model, hf_tokenizer, text_splitter

def safe_get_str(meta: dict, *keys, default: str = "") -> str:
    """
    Summary: Truy xuất giá trị lồng nhau và trả về dưới dạng chuỗi.
    - Nếu là str thì giữ nguyên.
    - Nếu là int/float/bool thì chuyển thành str.
    - Ngược lại hoặc không tồn tại thì trả default.
    """
    cur = meta
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if cur is None:
        return default
    if isinstance(cur, str):
        return cur
    if isinstance(cur, (int, float, bool)):
        return str(cur)
    return default

def safe_get_int(meta: dict, *keys, default: int = 0) -> int:
    """
    Summary: Truy xuất giá trị lồng nhau và ép về int nếu được.
    - Chuyển từ str nếu có thể.
    - Nếu là float hoặc bool thì ép về int.
    - Nếu không hợp lệ hoặc không tồn tại thì trả default.
    """
    cur = meta
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        if isinstance(cur, bool):  # tránh bool bị hiểu là int không mong muốn
            return int(cur)
        if isinstance(cur, (int, float)):
            return int(cur)
        if isinstance(cur, str):
            return int(cur.strip())
    except (ValueError, TypeError):
        pass
    return default

def safe_get_bool(meta: dict, *keys, default: bool = False) -> bool:
    """
    Summary: Truy xuất giá trị lồng nhau và quy về boolean.
    - Hỗ trợ bool trực tiếp, số (0=false, khác=true), và chuỗi như 'true','1','yes','no'.
    - Nếu không xác định được thì trả default.
    """
    cur = meta
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, bool):
        return cur
    if isinstance(cur, (int, float)):
        return bool(cur)
    if isinstance(cur, str):
        v = cur.strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            return True
        if v in ("false", "0", "no", "n", "f"):
            return False
    return default

def get_entry_type_by_text(metas, jp_text: str) -> str:
    """
    Summary: Tìm entry có text đúng bằng jp_text, trả về type top-level ('0' hoặc '1').
    Nếu không tìm thấy hoặc type không hợp lệ thì mặc định '1'.
    """
    for m in metas:
        if safe_get_str(m, "text") == jp_text:
            t = safe_get_str(m, "type")  # top-level
            if t in ("0", "1"):
                return t
            break
    return "1"

# def evaluate_bleu(
#     emb,
#     index: faiss.Index,
#     metas: List[dict],
#     llm,
#     cfg,
#     test_data: List[Tuple[str, str]],
# ) -> float:
#     """
#     Compute corpus-level BLEU over a test set.
#     test_data: List of (jp_sentence, reference_translation).
#     Returns BLEU score in [0,100].
#     """
#     smoothie = SmoothingFunction().method1
#     references, hypotheses = [], []

#     for jp_sentence, gold_ref in test_data:
#         # Dịch câu (đã bao gồm tách label segments)
#         pred = translate_one(emb, index, metas, llm, cfg, jp_sentence)

#         # Tokenize on whitespace
#         hyp_tokens = pred.strip().split()
#         ref_tokens = gold_ref.strip().split()

#         hypotheses.append(hyp_tokens)
#         # BLEU expects list of possible references per hypothesis
#         references.append([ref_tokens])

#     bleu = corpus_bleu(
#         references,
#         hypotheses,
#         weights=(0.25,0.25,0.25,0.25),
#         smoothing_function=smoothie
#     ) * 100.0

#     print(f"\n=== Corpus BLEU score: {bleu:.2f} ===")
#     return bleu