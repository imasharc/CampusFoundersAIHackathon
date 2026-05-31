# Raumdeuter — Fan Intelligence Pipeline

> **GDPR-compliant, real-time football fan feedback analysis.**  
> Multilingual (EN + DE) · Sentiment · PII masking · Topic classification

Built at the **CampusFounders AI Hackathon 2025** by Team 14 — Antoni Malinowski, Malak Abdallah, Omar Azlan, Abdallah Dinc.

---

## Two modes

### Local (full pipeline)

Run locally to get the complete 3-step pipeline with local ML inference:

```
Raw fan message
      │
      ▼
┌─────────────────────────────┐
│  01  XLM-RoBERTa  (local)  │  Multilingual sentiment scoring on the
│                             │  raw text — no data leaves the machine
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  02  Presidio      (local)  │  Detects & masks PII — names, emails,
│                             │  phones, IBANs, credit cards (EN + DE)
└─────────────────────────────┘
      │  (only masked text proceeds)
      ▼
┌─────────────────────────────┐
│  03  LLaMA 3.3 via Groq     │  Topic classification + reasoning
└─────────────────────────────┘
```

Sentiment is scored by RoBERTa **before** PII masking — so it sees the full original message for maximum accuracy. The score is passed to Groq as context; the LLM only classifies topics.

### Deployed / Railway (lightweight)

The Railway deployment strips PyTorch and the large spaCy models to keep the Docker image under 300 MB. Presidio still runs locally inside the container — PII never leaves. Groq handles both sentiment and topics in a single call.

```
Raw fan message
      │
      ▼
┌─────────────────────────────┐
│  01  Presidio      (local)  │  PII masking (spaCy sm models, ~30 MB)
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  02  LLaMA 3.3 via Groq     │  Sentiment + topics + reasoning
└─────────────────────────────┘
```

| | Local | Deployed |
|--|-------|----------|
| Sentiment engine | XLM-RoBERTa (local) | LLaMA 3.3 via Groq |
| spaCy models | `en/de_core_web_lg` (~1.6 GB) | `en/de_core_web_sm` (~30 MB) |
| PyTorch required | Yes | No |
| Docker image size | ~5 GB | ~300 MB |
| GDPR — PII leaves machine | Never | Never |

---

## Stack

| Component | Local | Deployed |
|-----------|-------|----------|
| FastAPI + uvicorn | ✓ | ✓ |
| XLM-RoBERTa (sentiment) | ✓ local inference | — |
| Microsoft Presidio + spaCy lg | ✓ | — |
| Microsoft Presidio + spaCy sm | — | ✓ |
| Groq API — LLaMA 3.3 70B | topics only | sentiment + topics |
| uv | ✓ | — |

---

## Prerequisites

| Tool | Min version | Check |
|------|------------|-------|
| Python | 3.11 | `python --version` |
| uv | any | `uv --version` |

**Install uv:**

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy BypassPolicy -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

### Local (full pipeline with RoBERTa)

### 1. Clone and install

```bash
git clone https://github.com/imasharc/CampusFoundersAIHackathon.git
cd CampusFoundersAIHackathon
uv sync
```

First run downloads PyTorch + Transformers (~2 GB).

### 2. Download spaCy models (one-time, ~1.6 GB)

```bash
uv run python -m spacy download en_core_web_lg
uv run python -m spacy download de_core_news_lg
```

### 3. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up — GitHub login works, no credit card needed
3. **API Keys → Create API Key** — copy the key (starts with `gsk_`)

### 4. Create `.env`

```env
GROQ_API_KEY=gsk_your_key_here
```

> `.env` is in `.gitignore` — never committed.

### 5. Run

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Wait for:
```
Loading RoBERTa...
Loading Presidio (EN + DE)...
All models ready.
Key loaded status: True
```

### 6. Open

```
http://localhost:8000
```

---

### Lightweight mode (matches the Railway deployment)

To run the same lightweight version locally — no PyTorch, small spaCy models:

```bash
# Switch to small spaCy models
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download de_core_news_sm

# Use the lightweight pipeline (pipeline_light.py or swap pipeline.py)
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Project structure

```
.
├── pipeline.py        # Full pipeline: RoBERTa → Presidio lg → Groq (local)
├── pipeline_light.py  # Lightweight: Presidio sm → Groq (deployed)
├── server.py          # FastAPI: /analyse/stream, /analyse, /health
├── index.html         # Single-file UI served at /
├── pyproject.toml     # Dependencies (uv)
├── Dockerfile         # Lightweight build for Railway
└── .env               # API keys — NOT committed
```

---

## API

### `POST /analyse/stream`
Server-Sent Events — streams results as steps complete.

```bash
curl -N -X POST http://localhost:8000/analyse/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "The atmosphere last night was electric!"}'
```

Events: `roberta_start` → `sentiment_done` → `done`

### `POST /analyse`
Synchronous — returns full JSON in one shot.

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"message": "Ticket prices are getting ridiculous."}'
```

```json
{
  "sentiment_score":  0.85,
  "sentiment_label":  "Negative",
  "key_topics":       ["Ticket Pricing"],
  "intent":           "complaint",
  "confidence":       0.92,
  "reasoning":        "Strong negative framing around pricing...",
  "pii_masked":       false,
  "masked_text":      "Ticket prices are getting ridiculous.",
  "pii_entities":     [],
  "processed_at":     "2025-05-22T10:00:00Z"
}
```

### `GET /health`
Returns `{"status": "ok"}`.

---

## Deploying to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**
3. Select the repo — Railway auto-detects the `Dockerfile`
4. **Settings → Variables** → add `GROQ_API_KEY=gsk_...`
5. Done — live URL provided automatically

---

## Running locally with a public URL (Cloudflare Tunnel)

```bash
# Terminal 1
uv run uvicorn server:app --host 127.0.0.1 --port 8000

# Terminal 2 — Windows
./cloudflared.exe tunnel --url http://localhost:8000
```

Gives a live `https://random-name.trycloudflare.com` URL. No account needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Key loaded status: False` | `.env` missing or key name wrong — must be `GROQ_API_KEY` |
| `401 Unauthorized` from Groq | Key expired — generate a new one at console.groq.com |
| `spacy model not found` | Run the `spacy download` commands from Step 2 |
| Port 8000 already in use | Add `--port 8001` to the uvicorn command |
| `uv` not recognised | Restart terminal after installing |

---

*CampusFounders AI Hackathon 2025*