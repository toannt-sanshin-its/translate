from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import json
import os

from config import Config
from translate_cli import load_resources, translate_one, get_device_label

app = FastAPI(
    title="Japanese→English Translation API",
    description="API for translating Japanese sentences to English using RAG + LLaMA.cpp",
    version="1.0.0"
)

# Load models at startup
cfg = Config()
emb, index, metas, llm = load_resources(cfg)
device = get_device_label(cfg)

class TranslateRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

@app.get("/translate")
def translate(text: str):
    """
    Translate a single Japanese sentence to English.
    """
    start = time.perf_counter()
    try:
        vi, src, score = translate_one(emb, index, metas, llm, cfg, text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.perf_counter() - start
    with open("translate_log.txt", "a", encoding="utf-8") as logf:
        logf.write(f"TIME: {round(elapsed, 4)}\tSCORE: {score:.4f}\tDEVICE: {device}\tJP: {text}\tEN: {vi}\n")
    return {
        "query": text,
        "translation": vi,
        "score": score,
        "device": device,
        "elapsed_s": round(elapsed, 4)
    }

@app.post("/batch_translate")
def batch_translate(req: BatchRequest):
    """
    Translate a batch of Japanese sentences to English.
    """
    results = []
    for text in req.texts:
        start = time.perf_counter()
        try:
            vi, src, score = translate_one(emb, index, metas, llm, cfg, text)
        except Exception as e:
            results.append({"query": text, "error": str(e)})
            continue
        elapsed = time.perf_counter() - start
        results.append({
            "query": text,
            "translation": vi,
            "score": score,
            "device": device,
            "elapsed_s": round(elapsed, 4)
        })
    return {"results": results}

# To run: uvicorn translate_api:app --host 0.0.0.0 --port 8000 --reload
