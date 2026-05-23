# Eval Harness

Run all guardrail evaluations in isolation against labeled datasets.

## Usage

```bash
python evals/run_evals.py
```

Results are saved to `evals/results/eval_report.txt`.

## Datasets

| File | Guardrail | Samples |
|------|-----------|---------|
| `injection_attacks.jsonl` | Input guard — injection detection | 40 |
| `pii_samples.jsonl` | PII redactor — entity detection | 30 |
| `grounding_samples.jsonl` | Grounding checker | 30 |
| `toxic_outputs.jsonl` | Toxicity filter | 30 |

## Notes

- Injection and toxicity evals run fully locally (no Ollama needed).
- Grounding checker eval requires Ollama to be running for embedding calls.
- Each guardrail is evaluated in isolation (not through the full graph) for speed.
