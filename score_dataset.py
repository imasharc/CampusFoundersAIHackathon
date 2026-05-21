# score_dataset.py
# Adds a numeric sentiment_score (0.0–1.0) column to your dataset using RoBERTa.
# The existing Positive/Neutral/Negative labels stay untouched.
# A score near 0 = very positive, near 1 = very negative.
#
# Run:  uv run python score_dataset.py
# Output: fan_dataset_200_scored.xlsx

import pandas as pd
from transformers import pipeline as hf_pipeline

DATASET_IN  = "fan_dataset_200_final.xlsx"
DATASET_OUT = "fan_dataset_200_scored.xlsx"
SHEET       = "Fan Messages"

print("Loading RoBERTa...")
sentiment = hf_pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    top_k=None, truncation=True, max_length=512,
)
print("Model ready.\n")

df = pd.read_excel(DATASET_IN, sheet_name=SHEET)
df.columns = df.columns.str.strip()

raw_col  = next(c for c in df.columns if "Raw Message" in c)
sent_col = next(c for c in df.columns if "Sentiment"    in c)

scores = []
for i, raw in enumerate(df[raw_col].astype(str)):
    result = sentiment(raw)[0]
    s      = {r["label"].lower(): r["score"] for r in result}
    neg    = s.get("negative", 0)
    neu    = s.get("neutral",  0)
    score  = round(neg + neu * 0.5, 4)
    scores.append(score)

    label = df[sent_col].iloc[i]
    predicted = (
        "Positive" if score <= 0.30 else
        "Negative" if score >= 0.48 else "Neutral"
    )
    match_mark = "✓" if predicted.lower() == str(label).strip().lower() else "✗"
    print(f"[{i+1:>3}/{len(df)}]  score={score:.3f}  pred={predicted:<8}  true={label:<8}  {match_mark}")

df["sentiment_score"] = scores
df.to_excel(DATASET_OUT, index=False)

# Quick accuracy summary
predicted_labels = [
    "Positive" if s <= 0.35 else "Negative" if s >= 0.65 else "Neutral"
    for s in scores
]
true_labels = df[sent_col].astype(str).str.strip().tolist()
correct = sum(p.lower() == t.lower() for p, t in zip(predicted_labels, true_labels))

print(f"\n{'='*50}")
print(f"Saved to:  {DATASET_OUT}")
print(f"New col:   sentiment_score  (0.0 = positive → 1.0 = negative)")
print(f"Accuracy:  {correct}/{len(df)} = {correct/len(df):.1%}")
print(f"{'='*50}")
print("\nNote: Use fan_dataset_200_scored.xlsx for future numeric evaluation.")