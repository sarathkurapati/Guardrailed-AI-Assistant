FROM python:3.11-slim

WORKDIR /app

# System deps for Presidio/spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Download spaCy model (required by Presidio)
RUN python -m spacy download en_core_web_lg

# Download toxic-bert at build time (no runtime surprise)
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='unitary/toxic-bert', top_k=None)"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
