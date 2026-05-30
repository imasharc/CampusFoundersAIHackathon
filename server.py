# server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import json
import asyncio
from pydantic import BaseModel
from pathlib import Path
import pipeline as pl
import os

app = FastAPI(title="Raumdeuter Fan Intelligence")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_key = os.environ.get("GROQ_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
print("Key loaded status:", bool(_key), flush=True)

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
        e1 = json.dumps({"step": "roberta_start"})
        yield "data: " + e1 + "\n\n"

        result = await asyncio.to_thread(pl.run_pipeline, req.message)

        e2 = json.dumps({
            "step": "sentiment_done",
            "sentiment_score": result.get("sentiment_score"),
            "sentiment_label": result.get("sentiment_label"),
            "pii_masked": result.get("pii_masked"),
        })
        yield "data: " + e2 + "\n\n"

        await asyncio.sleep(0.3)

        result["step"] = "done"
        yield "data: " + json.dumps(result) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok"}