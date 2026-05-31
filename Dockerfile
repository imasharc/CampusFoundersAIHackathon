FROM python:3.11-slim

WORKDIR /app

# System deps needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly with pip
RUN pip install --no-cache-dir \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.31.0" \
    "transformers>=4.40.0" \
    "torch>=2.1.0" \
    "presidio-analyzer>=2.2.0" \
    "presidio-anonymizer>=2.2.0" \
    "pydantic>=2.0.0" \
    "pandas>=2.0.0" \
    "openpyxl>=3.1.0"

# Download spaCy language models
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download de_core_news_lg

# Pre-cache RoBERTa weights into the image so first request is instant
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', model='cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual', top_k=None); \
print('RoBERTa weights cached.')"

# Copy application files
COPY pipeline.py server.py index.html ./

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]