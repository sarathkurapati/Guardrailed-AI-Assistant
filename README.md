# Guardrailed Enterprise AI Assistant

A production-oriented LLM pipeline that orchestrates safety, validation, and compliance guardrails around a RAG generation core — built with LangGraph, Llama 3.1 8B, and a fully local, zero-cost stack. Every decision in the pipeline is traceable via a per-session audit trail, and every metric in this README is real and reproducible by running a single command.

---

## Architecture

```
[START]
   │
   ▼
input_guard ──(blocked)────────────────────────► audit_logger ► [END]
   │(passed)
   ▼
retriever ► generator ► output_validator
                              │(fail, attempts < 3) ──► generator (cycle)
                              │(pass OR attempts == 3)
                              ▼
                        pii_redactor ► grounding_checker ► hitl_router
                                                               │(escalate) ► audit_logger ► [END] (202)
                                                               │(auto_respond)
                                                               ▼
                                                         audit_logger ► [END] (200)
```

### Node Summary

| Node | Responsibility |
|------|----------------|
| `input_guard` | Regex injection detection, LLM classifier, scope check, Presidio PII scan |
| `retriever` | Embed query → ChromaDB top-5 similarity search |
| `generator` | RAG prompt → Llama 3.1 8B, increment attempts |
| `output_validator` | toxic-bert, bias LLM check, length/empty check; drives retry cycle |
| `pii_redactor` | Presidio NER → replace with `<ENTITY_TYPE>` |
| `grounding_checker` | Extract claims → cosine similarity against retrieved docs |
| `hitl_router` | Confidence score → auto-respond or escalate to human queue |
| `audit_logger` | Write full audit trail JSON per session |

---

## Guardrail Performance

| Guardrail | Method | Eval F1 |
|-----------|--------|---------|
| Injection detection | Regex + LLM classifier | see `evals/results/` |
| PII redaction | Presidio NER | see `evals/results/` |
| Toxicity filtering | toxic-bert | see `evals/results/` |
| Grounding check | Embedding cosine similarity | see `evals/results/` |

Run `python evals/run_evals.py` to reproduce all numbers.

---

## Stack

| Concern | Tool |
|---------|------|
| LLM inference | Ollama + Llama 3.1 8B |
| Embeddings | nomic-embed-text via Ollama |
| Orchestration | LangGraph 0.2.x |
| LLM abstraction | LangChain 0.2.x |
| Vector store | ChromaDB (persistent local) |
| PII detection | Microsoft Presidio + spaCy `en_core_web_lg` |
| Toxicity classifier | `unitary/toxic-bert` (HuggingFace) |
| API | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |
| Observability | structlog (JSON logs) |

**All tools are free and run locally. No API keys or cloud services required.**

---

## How to Run

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Docker (optional, for containerised run)

### 1. Pull required models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
```

### 3. Ingest sample documents

```bash
python scripts/ingest.py
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API is now running at `http://localhost:8000`.

### 5. Send your first query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the annual leave policy?"}'
```

### 6. Docker (alternative)

```bash
docker compose up --build
```

Ollama must be running on the host. The container reaches it via `host.docker.internal:11434`.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Submit a query through the full pipeline |
| `/audit/{session_id}` | GET | Retrieve full audit trail for a session |
| `/health` | GET | Service health including Ollama and ChromaDB status |
| `/hitl/queue` | GET | List sessions pending human review |
| `/hitl/resolve/{session_id}` | POST | Submit approved response for a queued session |

### Response status values

- `ok` — pipeline completed, response returned (HTTP 200)
- `blocked` — input guard rejected the query (HTTP 200)
- `pending_review` — escalated to human queue (HTTP 202)

---

## Run Evals

```bash
python evals/run_evals.py
```

Results are saved to `evals/results/eval_report.txt`.

---

## Run Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

Integration tests that require Ollama are skipped automatically in CI (`CI=true`).

---

## Design Decisions

- **Cyclic graph over linear retry**: LangGraph's conditional edges let `output_validator` send failing responses back to `generator` for up to 3 attempts before escalating. A linear pipeline would require external orchestration logic; the graph captures this natively.

- **Presidio over regex for PII**: Regex cannot reliably detect names, IBAN codes, or IP addresses in free text. Presidio's NER-based approach with spaCy `en_core_web_lg` achieves higher recall on structured entity types.

- **Embedding cosine similarity for grounding**: Semantic similarity correctly handles paraphrase and synonymy — a claim that "employees get 20 vacation days" is grounded by a document saying "20 days of paid annual leave" even without exact string overlap.

- **MemorySaver checkpointer for HITL**: LangGraph's `MemorySaver` persists graph state keyed on `session_id`, enabling the HITL queue to resume a conversation after human review without re-running the pipeline.

- **All thresholds in config**: Every numeric threshold (`toxicity_threshold`, `grounding_similarity_threshold`, etc.) is an environment variable via `pydantic-settings`. No magic numbers in node code — tuning doesn't require code changes.

---

## Limitations and Future Work

- **Language**: Presidio and toxic-bert are English-only. Multi-language support would require per-language Presidio models and a multilingual toxicity classifier.
- **HITL queue**: File-based JSON queue simulates a review inbox. Production would use PostgreSQL + a notification system (e.g. Slack webhook).
- **Grounding threshold is heuristic**: The 0.75 cosine similarity threshold was chosen empirically. A calibrated threshold per domain would improve precision.
- **Ollama cold-start**: Llama 3.1 8B takes 5–15 seconds to load on first call. Subsequent calls are fast (~2–5s on CPU). Document expected response times to users.
- **toxic-bert false positives on technical language**: Terms like "kill process" or "attack surface" may trigger false positives. The toxicity score is used as a soft confidence penalty below 0.8, not a hard block.
- **No authentication**: The API has no auth layer. Production deployment requires at minimum an API key header or OAuth2.
