"""
Đọc biến môi trường & cung cấp hằng số cấu hình cho dự án dịch JP→VI (RAG).

- Tự động nạp .env (nếu có)
- Hàm getenv_* an toàn (tránh lỗi khi trùng tên biến hệ thống như TEMP trên Windows)
- Gom cấu hình vào dataclass Config
- Có hàm validate() để cảnh báo thiếu file/path
"""

from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass  # nếu không có dotenv, bỏ qua

# ----------------- Helpers ----------------- #
def getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default

def getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default

def getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default

def getenv_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

def expand_path(p: str) -> str:
    """Mở rộng ~, biến môi trường, chuyển về đường dẫn tuyệt đối chuẩn hóa."""
    return str(Path(os.path.expandvars(os.path.expanduser(p))).resolve())

# ----------------- Dataclass ----------------- #
@dataclass
class Config:
    # --- Embedding & Retrieval --- #
    EMB_MODEL: str = getenv_str("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    INDEX_PATH: str = expand_path(getenv_str("INDEX_PATH", "./indexes/faiss.index"))
    META_PATH:  str = expand_path(getenv_str("META_PATH",  "./indexes/meta.pkl"))

    TOP_K: int = getenv_int("TOP_K", 3)
    MIN_SCORE: int = getenv_int("MIN_SCORE", 0.7)
    NORMALIZE_EMB: bool = getenv_bool("NORMALIZE_EMB", True)

    # --- Runtime / Device --- #
    DEVICE: str = getenv_str("DEVICE", "cpu")  # "cpu" / "cuda"
    THREADS: int = getenv_int("THREADS", os.cpu_count() or 4)

    # --- Generation model (GGUF) --- #
    # Dùng cho llama.cpp hoặc ctransformers
    # MODEL_PATH: str = expand_path(getenv_str("MODEL_PATH", "models/vinallama-7b-chat_q5_0.gguf"))
    # MODEL_PATH: str = expand_path(getenv_str("MODEL_PATH", "models/vicuna-7b-v1.5.Q4_K_S.gguf"))
    MODEL_PATH: str = expand_path(getenv_str("MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q8_0.gguf"))
    
    N_CTX: int = getenv_int("N_CTX", 4096)

    # llama.cpp-specific
    N_GPU_LAYERS: int = getenv_int("N_GPU_LAYERS", 0)  # 0 = CPU

    # --- Decode params --- #
    # Dùng GEN_TEMP để tránh xung đột với biến hệ thống TEMP (Windows)
    TEMPERATURE: float = getenv_float("GEN_TEMP", 0.2)
    MAX_TOKENS: int = getenv_int("MAX_TOKENS", 256)

    # --- Prompt templates --- #
    SYSTEM_PROMPT = (
        "You are a concise Japanese → English translation expert. \n"
        # "Translate each sentence literally and concisely, producing only the exact English equivalent—no additional commentary or extra words."
        # "Translate each Japanese phrase or sentence literally and concisely.\n"
        # "Respond **only** with the exact English equivalent—no prefixes, no markup, no extra words, no explanations."
        # "ANY text between 「 and 」 MUST remain unchanged.\n"
        # "ANY text between [ and ] MUST remain unchanged.\n"
        "Translate literally and concisely. \nRespond only with the exact English equivalent—no prefixes, no markup, no extra words, no explanations."
    )

    USER_PROMPT_TEMPLATE = getenv_str(
        "USER_PROMPT_TEMPLATE",
        # Đổi nhãn để nhấn mạnh đây chỉ là ví dụ style, không phải nội dung cần copy:
        "Relevant examples (style only):\n"
        # "- Example:\n"
        "{context}\n\n"
        "-------------\n"
        # "Original sentence:\n"
        "Now translate literally and concisely:\n"
        "Q:{query}\n"
        "A:\n\n"
        "Return only A\n"
        "-------------\n"
        # "Aim for accuracy and clarity in your response"
    )

    # Template khi có context
    TEMPLATE_WITH_CTX: str = getenv_str(
        "USER_PROMPT_TEMPLATE",
        "Relevant examples (style only):\n"
        "{context}\n\n"
        "Original sentence:\n{query}\n\n"
        "Please translate the above sentence literally and concisely."
    )

    # Template khi context trống
    TEMPLATE_NO_CTX: str = getenv_str(
        "USER_PROMPT_TEMPLATE",
        "Original sentence:\n{query}\n\n"
        "Translate _only_ this sentence into English."
    )

    # --- Logging / Misc --- #
    LOG_LEVEL: str = getenv_str("LOG_LEVEL", "INFO")

    # -------------- Methods -------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self, strict: bool = False) -> None:
        """
        Kiểm tra sự tồn tại một số file quan trọng. Nếu strict=True, raise lỗi khi thiếu.
        """
        missing = []
        for path_key in ["INDEX_PATH", "META_PATH", "MODEL_PATH"]:
            p = getattr(self, path_key)
            if not Path(p).exists():
                missing.append((path_key, p))

        if missing:
            msg_lines = ["⚠️ Thiếu file/đường dẫn:"]
            for k, p in missing:
                msg_lines.append(f"  - {k}: {p}")
            msg = "\n".join(msg_lines)

            if strict:
                raise FileNotFoundError(msg)
            else:
                print(msg)

# ----------------- Global singleton (optional) ----------------- #
# Bạn có thể dùng trực tiếp:
#   from config import CFG
#   ... rồi CFG.validate()
CFG = Config()
