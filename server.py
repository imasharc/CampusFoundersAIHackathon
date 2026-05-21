# pipeline.py
import os, json, datetime
import requests
from dotenv import load_dotenv

load_dotenv()

# ── load once at startup, not per request ─────────────────────────────────────
from transformers import pipeline as hf_pipeline
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

print("Loading RoBERTa...")
_sentiment = hf_pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    top_k=None, truncation=True, max_length=512,
)

print("Loading Presidio (EN + DE)...")
_provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
        {"lang_code": "de", "model_name": "de_core_news_lg"},
    ],
})
_analyzer   = AnalyzerEngine(nlp_engine=_provider.create_engine())
_anonymizer = AnonymizerEngine()

print("All models ready.")

# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT = """
Examples:
"[PERSON] performance was shocking, same tactics every game" → topics: ["Team Performance","Tactics"] intent: complaint
"When do season tickets go on sale?" → topics: ["Season Tickets","Ticket Booking"] intent: inquiry  
"Atmosphere last night was electric, best match in years" → topics: ["Atmosphere","Stadium Experience"] intent: praise
"Ich war begeistert aber die Preise sind unverschämt hoch" → topics: ["Atmosphere","Ticket Pricing"] intent: complaint
"New striker is incredible, totally transformed us going forward" → topics: ["Player Performance","Team Performance"] intent: praise
"""

TOPICS = [
    "Team Performance","Player Performance","Tactics","Match Outcome","Refereeing",
    "Ticket Pricing","Ticket Booking","Season Tickets","Merchandise",
    "Stadium Experience","Atmosphere","Food & Drink","Accessibility",
    "Club Communication","Transfer News","Manager","Youth Academy",
    "Excitement","Frustration","Praise",
]

def run_pipeline(raw_message: str) -> dict:

    # ── 1. RoBERTa on raw text (local, sees full context before masking) ──────
    scores = {s["label"].lower(): s["score"] for s in _sentiment(raw_message)[0]}
    neg = scores.get("negative", 0)
    neu = scores.get("neutral",  0)
    sentiment_score = round(neg + neu * 0.5, 4)
    sentiment_label = (
        "Positive" if sentiment_score <= 0.35 else
        "Negative" if sentiment_score >= 0.65 else "Neutral"
    )

    # ── 2. Presidio on raw text (local, strips PII) ───────────────────────────
    lang = "de" if _is_german(raw_message) else "en"
    results = _analyzer.analyze(
        text=raw_message, language=lang,
        entities=["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","LOCATION"],
        score_threshold=0.4,
    )
    masked_text = _anonymizer.anonymize(raw_message, results).text if results else raw_message

    # ── 3. Groq (LLaMA 3.3) on masked text + sentiment score ─────────────────────
    llm = _groq_call(masked_text, sentiment_label, sentiment_score)

    return {
        "sentiment_score":  sentiment_score,
        "sentiment_label":  sentiment_label,
        "reasoning":        llm.get("reasoning", ""),
        "key_topics":       llm.get("key_topics", []),
        "intent":           llm.get("intent", "other"),
        "topic":            llm.get("key_topics", ["Other"])[0] if llm.get("key_topics") else "Other",
        "confidence":       llm.get("confidence", 0.0),
        "pii_masked":       len(results) > 0,
        "processed_at":     datetime.datetime.utcnow().isoformat() + "Z",
    }


def _groq_call(masked_text, sentiment_label, sentiment_score):
    prompt = f"""You are a football fan message classifier.

Pre-scored sentiment: {sentiment_label} ({sentiment_score:.2f} where 0=positive 1=negative).
Trust this score. Your job is topics + reasoning only.

{FEW_SHOT}

Message: "{masked_text}"
Topics to choose from: {TOPICS}

Return ONLY valid JSON, nothing else:
{{
  "key_topics": ["2-4 topics from the list above"],
  "intent": "complaint|inquiry|praise|other",
  "confidence": 0.0-1.0,
  "reasoning": "1-2 sentences: what phrases drove the sentiment, what is the comment about"
}}"""

    # 1. Strip hidden spaces/newlines from the .env file that corrupt HTTP headers
    api_key = os.environ.get('GROQ_API_KEY', '').strip()

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 200,
        },
        timeout=15,
    )
    
    # Print the actual API error instead of crashing blindly
    if not resp.ok:
        print(f"\n❌ GROQ API ERROR {resp.status_code} ❌", flush=True)
        print(resp.text, flush=True)
        print("-----------------------------------\n", flush=True)
        resp.raise_for_status()

    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"key_topics": [], "intent": "other", "confidence": 0.0, "reasoning": ""}

def _is_german(text: str) -> bool:
    de = {"ich","bin","habe","nicht","das","ist","und","mit","für","eine","mein","war","sie","wir"}
    return len(set(text.lower().split()) & de) >= 2