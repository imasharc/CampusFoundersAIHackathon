# evaluate.py — run the pipeline against your labeled dataset
# This is how your dataset is useful WITHOUT training anything
#
# Run:  uv run python evaluate.py
#
# Output: accuracy numbers + a CSV of every prediction vs ground truth

import pandas as pd
import time
from pipeline import run_pipeline

# Load your dataset
df = pd.read_excel("fan_dataset_200_final.xlsx", sheet_name="Fan Messages")
df.columns = df.columns.str.strip()

raw_col  = [c for c in df.columns if "Raw Message" in c][0]
sent_col = [c for c in df.columns if "Sentiment" in c][0]

results = []
correct_sentiment = 0
total = 0

for i, row in df.iterrows():
    raw  = str(row[raw_col])
    true_sentiment = str(row[sent_col]).strip()

    print(f"[{i+1}/{len(df)}] Processing...")

    try:
        out = run_pipeline(raw)
        pred_sentiment = out["sentiment_label"]
        match = pred_sentiment.lower() == true_sentiment.lower()

        if match:
            correct_sentiment += 1
        total += 1

        results.append({
            "raw_message":      raw[:100] + "...",
            "true_sentiment":   true_sentiment,
            "pred_sentiment":   pred_sentiment,
            "sentiment_score":  out["sentiment_score"],
            "sentiment_match":  "✓" if match else "✗",
            "key_topics":       ", ".join(out["key_topics"]),
            "intent":           out["intent"],
            "reasoning":        out["reasoning"][:120],
            "pii_masked":       out["pii_masked"],
        })

        time.sleep(0.3)  # avoid Fireworks rate limit

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"raw_message": raw[:80], "error": str(e)})

# Save results
out_df = pd.DataFrame(results)
out_df.to_excel("pipeline_evaluation.xlsx", index=False)

# Print summary
accuracy = correct_sentiment / total if total > 0 else 0
print(f"\n{'='*50}")
print(f"Sentiment accuracy: {accuracy:.1%}  ({correct_sentiment}/{total} correct)")
print(f"Results saved to: pipeline_evaluation.xlsx")
print(f"{'='*50}")
print("\nWrong predictions:")
wrong = out_df[out_df.get("sentiment_match") == "✗"]
for _, r in wrong.iterrows():
    print(f"  TRUE={r['true_sentiment']} PRED={r['pred_sentiment']}  {r['raw_message'][:60]}")
