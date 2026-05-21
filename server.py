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
print("Key loaded status:", bool(os.environ.get('FIREWORKS_API_KEY')), flush=True)

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
        # 1. Fire the first event to tell the UI that PII masking is complete
        yield f"data: {json.dumps({'step': 'pii_done'})}\n\n"
        
        # 2. Run your ML pipeline in a background thread so it doesn't block the server
        result = await asyncio.to_thread(pl.run_pipeline, req.message)
        
        # 3. Fire the sentiment event using the real data we just got back
        yield f"data: {json.dumps({
            'step': 'sentiment_done', 
            'sentiment_score': result.get('sentiment_score'), 
            'sentiment_label': result.get('sentiment_label')
        })}\n\n"
        
        # (Optional) Small delay so the final UI step doesn't flash instantly
        await asyncio.sleep(0.3)
        
        # 4. Fire the 'done' event with the full payload to render the final dashboard
        result['step'] = 'done'
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok"}
