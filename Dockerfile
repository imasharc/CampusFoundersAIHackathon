FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies — no PyTorch, no Transformers, tiny image
RUN pip install --no-cache-dir \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.31.0" \
    "presidio-analyzer>=2.2.0" \
    "presidio-anonymizer>=2.2.0" \
    "pydantic>=2.0.0"

# Small spaCy models (~15MB each vs ~800MB for lg)
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm

COPY pipeline.py server.py index.html ./

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]