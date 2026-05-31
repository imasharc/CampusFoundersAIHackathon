# pipeline.py
import os, json, datetime
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Presidio with small spaCy models (no PyTorch) ────────────────────────────
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern as PPattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

print("Loading Presidio (EN + DE, small models)...")
_provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"},
        {"lang_code": "de", "model_name": "de_core_news_sm"},
    ],
})
_analyzer   = AnalyzerEngine(nlp_engine=_provider.create_engine())
_anonymizer = AnonymizerEngine()

# ── Custom IBAN recognizer ────────────────────────────────────────────────────
_iban_pattern = PPattern(
    name="iban",
    regex=r'\b[A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b',
    score=0.85,
)
for _lang in ("en", "de"):
    _analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="IBAN_CODE",
        patterns=[_iban_pattern],
        supported_language=_lang,
    ))

print("All models ready.")

# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT = """
Examples:
"<PERSON> performance was shocking, same tactics every game" → sentiment: Negative, topics: ["Team Performance","Tactics"] intent: complaint
"When do season tickets go on sale?" → sentiment: Neutral, topics: ["Season Tickets","Ticket Booking"] intent: inquiry
"Atmosphere last night was electric, best match in years" → sentiment: Positive, topics: ["Atmosphere","Stadium Experience"] intent: praise
"Ich war begeistert aber die Preise sind unverschämt hoch" → sentiment: Negative, topics: ["Atmosphere","Ticket Pricing"] intent: complaint
"New striker is incredible, totally transformed us going forward" → sentiment: Positive, topics: ["Player Performance","Team Performance"] intent: praise
"""

TOPICS = [
    "Team Performance","Player Performance","Tactics","Match Outcome","Refereeing",
    "Ticket Pricing","Ticket Booking","Season Tickets","Merchandise",
    "Stadium Experience","Atmosphere","Food & Drink","Accessibility",
    "Club Communication","Transfer News","Manager","Youth Academy",
    "Excitement","Frustration","Praise",
]

def run_pipeline(raw_message: str) -> dict:

    # ── 1. Presidio: strip PII locally before anything goes to the cloud ──────
    lang = "de" if _is_german(raw_message) else "en"
    pii_results = _analyzer.analyze(
        text=raw_message, language=lang,
        entities=["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","LOCATION","IBAN_CODE","CREDIT_CARD"],
        score_threshold=0.4,
    )
    masked_text = _anonymizer.anonymize(raw_message, pii_results).text if pii_results else raw_message

    # ── 2. Single Groq call: sentiment + topics + reasoning on masked text ────
    result = _groq_call(masked_text)

    # Normalise sentiment score to 0–1 (0=positive, 1=negative) for UI gauge
    label = result.get("sentiment", "Neutral")
    score_map = {"Positive": 0.10, "Neutral": 0.50, "Negative": 0.85}
    sentiment_score = score_map.get(label, 0.50)

    return {
        "sentiment_score":  sentiment_score,
        "sentiment_label":  label,
        "reasoning":        result.get("reasoning", ""),
        "key_topics":       result.get("key_topics", []),
        "intent":           result.get("intent", "other"),
        "topic":            result.get("key_topics", ["Other"])[0] if result.get("key_topics") else "Other",
        "confidence":       result.get("confidence", 0.0),
        "pii_masked":       len(pii_results) > 0,
        "masked_text":      masked_text,
        "pii_entities":     [{"type": r.entity_type, "start": r.start, "end": r.end} for r in pii_results],
        "processed_at":     datetime.datetime.utcnow().isoformat() + "Z",
    }


def _groq_call(masked_text: str) -> dict:
    prompt = f"""You are a football fan message classifier. Analyse the message and return ONLY valid JSON.

{FEW_SHOT}

Message: "{masked_text}"
Topics to choose from: {TOPICS}

Return ONLY valid JSON, nothing else:
{{
  "sentiment": "Positive|Neutral|Negative",
  "key_topics": ["2-4 topics from the list above"],
  "intent": "complaint|inquiry|praise|other",
  "confidence": 0.0-1.0,
  "reasoning": "1-2 sentences: what phrases drove the sentiment and what the comment is about"
}}"""

    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 200,
        },
        timeout=20,
    )

    if not resp.ok:
        print(f"\n❌ GROQ API ERROR {resp.status_code} ❌", flush=True)
        print(resp.text, flush=True)
        resp.raise_for_status()

    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"sentiment": "Neutral", "key_topics": [], "intent": "other",
                "confidence": 0.0, "reasoning": ""}


def _is_german(text: str) -> bool:
    de = {"ich","bin","habe","nicht","das","ist","und","mit","für","eine","mein","war","sie","wir"}
    return len(set(text.lower().split()) & de) >= 2