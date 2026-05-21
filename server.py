# server.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import json
import asyncio
import pandas as pd
from pydantic import BaseModel
from pathlib import Path
import pipeline as pl
import os

app = FastAPI(title="Raumdeuter Fan Intelligence")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Prefer GROQ_API_KEY, fall back to FIREWORKS_API_KEY for backwards compat
_key = os.environ.get("GROQ_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
print("Key loaded status:", bool(_key), flush=True)

DATASET_PATH = "fan_dataset_200_scored.xlsx"

class Request(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTMLResponse(Path("index.html").read_text(encoding="utf-8"))

@app.post("/analyse")
def analyse(req: Request):
    try:
        return pl.run_pipeline(req.message)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/analyse/stream")
async def analyse_stream(req: Request):
    async def event_generator():
        # 1. Signal that RoBERTa is starting
        e1 = json.dumps({"step": "roberta_start"})
        yield "data: " + e1 + "\n\n"

        # 2. Run the full pipeline in a thread (RoBERTa → Presidio → Groq)
        result = await asyncio.to_thread(pl.run_pipeline, req.message)

        # 3. Sentiment + PII done — fire scores so the UI can render the gauge
        e2 = json.dumps({
            "step": "sentiment_done",
            "sentiment_score": result.get("sentiment_score"),
            "sentiment_label": result.get("sentiment_label"),
            "pii_masked": result.get("pii_masked"),
        })
        yield "data: " + e2 + "\n\n"

        # 4. Short pause so the final panel doesn't flash in instantly
        await asyncio.sleep(0.3)

        # 5. Full payload — triggers complete result render
        result["step"] = "done"
        yield "data: " + json.dumps(result) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/evaluate/quick")
async def evaluate_quick(n: int = Query(default=20, ge=5, le=100)):
    """
    Run RoBERTa-only sentiment evaluation on n messages from the dataset.
    Fast — no LLM calls, purely local inference. Returns accuracy breakdown.
    """
    if not Path(DATASET_PATH).exists():
        raise HTTPException(
            404,
            detail=f"Dataset not found. Place '{DATASET_PATH}' in the project root."
        )

    df = pd.read_excel(DATASET_PATH, sheet_name="Fan Messages")
    df.columns = df.columns.str.strip()

    try:
        raw_col  = next(c for c in df.columns if "Raw Message" in c)
        sent_col = next(c for c in df.columns if "Sentiment" in c)
    except StopIteration:
        raise HTTPException(500, detail="Expected columns 'Raw Message' and 'Sentiment' not found.")

    sample = df.sample(min(n, len(df)), random_state=42)

    def run_eval():
        correct = 0
        results = []
        label_map = {"positive": 0, "neutral": 0, "negative": 0}
        for _, row in sample.iterrows():
            raw        = str(row[raw_col])
            true_label = str(row[sent_col]).strip().lower()

            scores = {s["label"].lower(): s["score"] for s in pl._sentiment(raw)[0]}
            neg    = scores.get("negative", 0)
            neu    = scores.get("neutral",  0)
            score  = round(neg + neu * 0.5, 4)
            pred   = (
                "positive" if score <= 0.35 else
                "negative" if score >= 0.65 else "neutral"
            )

            match = pred == true_label
            if match:
                correct += 1
            label_map[true_label] = label_map.get(true_label, 0) + 1

            results.append({
                "text":       raw[:90] + ("…" if len(raw) > 90 else ""),
                "true":       true_label.capitalize(),
                "pred":       pred.capitalize(),
                "score":      score,
                "match":      match,
            })

        accuracy = round(correct / len(sample), 3) if sample.shape[0] > 0 else 0
        wrong    = [r for r in results if not r["match"]]
        return {
            "accuracy":         accuracy,
            "correct":          correct,
            "total":            len(sample),
            "label_breakdown":  label_map,
            "wrong_predictions": wrong[:10],   # cap at 10 for UI
            "all_results":       results,
        }

    data = await asyncio.to_thread(run_eval)
    return data


@app.get("/health")
def health():
    return {"status": "ok"}