# Raumdeuter — Fan Intelligence Pipeline

> **GDPR-compliant, real-time football fan feedback analysis.**  
> Multilingual (EN + DE) · Sentiment · PII masking · Topic classification

---

## How it works

Three steps run in sequence every time a message is submitted:

```
Raw fan message
      │
      ▼
┌─────────────────────────────┐
│  01  XLM-RoBERTa  (local)  │  Scores sentiment on the full raw text
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  02  Presidio      (local)  │  Detects & masks PII — names, emails,
│                             │  phone numbers, locations
└─────────────────────────────┘
      │  (only masked text proceeds)
      ▼
┌─────────────────────────────┐
│  03  LLaMA 3.3 via Groq     │  Classifies topics + generates reasoning
└─────────────────────────────┘
      │
      ▼
  JSON result
```

**PII never leaves the machine.** RoBERTa and Presidio run entirely locally. Only the masked text reaches the cloud LLM.

---

## Stack

| Component | Role |
|-----------|------|
| FastAPI + uvicorn | Backend server, serves the UI |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | Local sentiment scoring (EN + DE) |
| Microsoft Presidio + spaCy | Local PII detection and anonymisation |
| Groq API — LLaMA 3.3 70B | Topic classification and reasoning |
| uv | Python dependency + environment management |

---

## Prerequisites

| Tool | Min version | Install check |
|------|------------|---------------|
| Python | 3.11 | `python --version` |
| uv | any | `uv --version` |
| Git | any | `git --version` |

**Install uv:**

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy BypassPolicy -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installing.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/CampusFoundersAIHackathon.git
cd CampusFoundersAIHackathon

uv sync
```

First run downloads PyTorch + Transformers (~2 GB). Get a coffee.

### 2. Download spaCy language models (one-time, ~800 MB)

```bash
uv run python -m spacy download en_core_web_lg
uv run python -m spacy download de_core_news_lg
```

### 3. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up — GitHub login works, no credit card needed
3. **API Keys → Create API Key** — copy the key (starts with `gsk_`)

### 4. Create `.env` in the project root

```env
GROQ_API_KEY=gsk_your_key_here
```

> `.env` is in `.gitignore` — it will never be committed.

### 5. Run the server

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Wait for all three lines before opening the browser:

```
Loading RoBERTa...
Loading Presidio (EN + DE)...
All models ready.
Key loaded status: True
```

First startup downloads RoBERTa weights (~500 MB). Subsequent starts are fast.

### 6. Open the app

```
http://localhost:8000
```

---

## Project structure

```
.
├── pipeline.py                   # Core pipeline: RoBERTa → Presidio → Groq
├── server.py                     # FastAPI: /analyse/stream, /evaluate/quick, /health
├── index.html                    # Single-file UI (served by FastAPI at /)
├── evaluate.py                   # Full batch evaluation against the dataset
├── score_dataset.py              # Adds numeric 0–1 scores to the Excel dataset
├── pyproject.toml                # Dependencies managed by uv
├── .env                          # API keys — NOT committed
├── fan_dataset_200_final.xlsx    # Raw labeled dataset (200 messages, EN + DE)
└── fan_dataset_200_scored.xlsx   # Dataset + RoBERTa sentiment_score column
```

---

## API reference

### `POST /analyse/stream`
Streams results via Server-Sent Events as each pipeline step completes.

```bash
curl -N -X POST http://localhost:8000/analyse/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "The atmosphere last night was electric!"}'
```

Events in order: `roberta_start` → `sentiment_done` → `done`

### `POST /analyse`
Synchronous — returns full JSON result in one shot.

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"message": "Ticket prices are getting ridiculous."}'
```

```json
{
  "sentiment_score": 0.81,
  "sentiment_label": "Negative",
  "key_topics": ["Ticket Pricing"],
  "intent": "complaint",
  "confidence": 0.92,
  "reasoning": "Strong negative framing around pricing...",
  "pii_masked": false,
  "processed_at": "2025-05-21T10:00:00Z"
}
```

### `GET /evaluate/quick?n=20`
Runs local RoBERTa evaluation on `n` messages from the dataset. No Groq calls — fast.

### `GET /health`
Returns `{"status": "ok"}`.

---

## Exposing publicly for demos

Download [cloudflared](https://github.com/cloudflare/cloudflared/releases/latest).

**Terminal 1:**
```bash
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

**Terminal 2:**
```bash
# Windows
.\cloudflared.exe tunnel --url http://localhost:8000

# Mac / Linux
./cloudflared tunnel --url http://localhost:8000
```

You get a live `https://random-name.trycloudflare.com` URL instantly. No account needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Key loaded status: False` | `.env` is missing or key is named wrong — must be exactly `GROQ_API_KEY` |
| `401 Unauthorized` from Groq | API key is wrong or expired — generate a new one at console.groq.com |
| `spacy model not found` | Run the two `spacy download` commands from Step 2 |
| Port 8000 already in use | Add `--port 8001` to the uvicorn command |
| `uv` not recognised | Restart terminal after installing, or re-run the install command |
| `Sheet1 not found` / `Fan Messages not found` | Check your dataset file is in the project root and named exactly `fan_dataset_200_scored.xlsx` |
| Server starts but returns 500 | Check terminal output — usually a missing `.env` or spaCy model |

---

## Dataset

`fan_dataset_200_scored.xlsx` — 200 labeled fan messages, English and German.

| Column | Description |
|--------|-------------|
| `Raw Message (with PII)` | Original text with realistic PII |
| `Masked A` | Pseudo-token masking (e.g. `USR_BF25`, `EML_2C17`) |
| `Masked B` | Fake persona masking (e.g. `Riley Shaw`) |
| `Topic` | Ground truth topic labels |
| `Sentiment` | Ground truth: Positive / Neutral / Negative |
| `sentiment_score` | Numeric 0–1 score from RoBERTa (0 = positive, 1 = negative) |

---

*CampusFounders AI Hackathon 2025*