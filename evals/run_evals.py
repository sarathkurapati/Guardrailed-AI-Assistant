"""
Eval harness for all 4 guardrails.
Runs each guardrail in isolation against its labeled dataset.

Usage: python evals/run_evals.py
Results saved to: evals/results/eval_report.txt
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.state import make_initial_state
from app.graph.nodes.input_guard import input_guard_node
from app.services.presidio_engine import analyze_text
from app.services.toxicity import score_toxicity
from app.config import settings
from app.utils.logging import configure_logging

configure_logging()

DATASETS = Path(__file__).parent / "datasets"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def precision_recall_f1(tp, fp, fn) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def eval_injection_detection() -> dict:
    samples = load_jsonl(DATASETS / "injection_attacks.jsonl")
    tp = fp = tn = fn = 0

    for s in samples:
        state = make_initial_state(s["query"], "eval-test")
        result = input_guard_node(state)
        predicted_injection = not result["input_guard_passed"]
        actual_injection = s["label"] == "injection"

        if actual_injection and predicted_injection:
            tp += 1
        elif not actual_injection and predicted_injection:
            fp += 1
        elif not actual_injection and not predicted_injection:
            tn += 1
        else:
            fn += 1

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "guardrail": "INPUT GUARD (Injection Detection)",
        "samples": len(samples),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def eval_pii_redaction() -> dict:
    samples = load_jsonl(DATASETS / "pii_samples.jsonl")
    tp = fp = tn = fn = 0

    for s in samples:
        results = analyze_text(s["text"])
        predicted_pii = len(results) > 0
        actual_pii = s["contains_pii"]

        if actual_pii and predicted_pii:
            tp += 1
        elif not actual_pii and predicted_pii:
            fp += 1
        elif not actual_pii and not predicted_pii:
            tn += 1
        else:
            fn += 1

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "guardrail": "PII REDACTOR",
        "samples": len(samples),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def eval_toxicity_filter() -> dict:
    samples = load_jsonl(DATASETS / "toxic_outputs.jsonl")
    tp = fp = tn = fn = 0

    for s in samples:
        score = score_toxicity(s["text"])
        predicted_toxic = score > settings.toxicity_threshold
        actual_toxic = s["is_toxic"]

        if actual_toxic and predicted_toxic:
            tp += 1
        elif not actual_toxic and predicted_toxic:
            fp += 1
        elif not actual_toxic and not predicted_toxic:
            tn += 1
        else:
            fn += 1

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "guardrail": "TOXICITY FILTER",
        "samples": len(samples),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def eval_grounding_checker() -> dict:
    import numpy as np
    from app.services.embeddings import get_embeddings

    samples = load_jsonl(DATASETS / "grounding_samples.jsonl")
    emb = get_embeddings()
    tp = fp = tn = fn = 0

    for s in samples:
        response = s["response"]
        docs_texts = s["docs"]

        # Treat as grounded if response is a refusal
        if "i don't have enough information" in response.lower():
            predicted_grounded = True
        elif not docs_texts:
            predicted_grounded = False
        else:
            # Check if any doc has cosine similarity >= threshold for any sentence
            try:
                doc_vecs = emb.embed_documents(docs_texts)
                resp_vec = emb.embed_query(response)

                def cosine(a, b):
                    a, b = np.array(a), np.array(b)
                    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

                max_sim = max(cosine(resp_vec, dv) for dv in doc_vecs)
                predicted_grounded = max_sim >= settings.grounding_similarity_threshold
            except Exception:
                predicted_grounded = True  # fail open

        actual_grounded = s["grounded"]

        if actual_grounded and predicted_grounded:
            tp += 1
        elif not actual_grounded and predicted_grounded:
            fp += 1
        elif not actual_grounded and not predicted_grounded:
            tn += 1
        else:
            fn += 1

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "guardrail": "GROUNDING CHECKER",
        "samples": len(samples),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def format_report(results: list[dict]) -> str:
    lines = ["=== EVAL REPORT ===", f"Run at: {datetime.now(timezone.utc).isoformat()}", ""]
    for r in results:
        lines.append(r["guardrail"])
        lines.append(f"  Samples tested  : {r['samples']}")
        lines.append(f"  True Positives  : {r['tp']}")
        lines.append(f"  False Positives : {r['fp']}")
        lines.append(f"  True Negatives  : {r['tn']}")
        lines.append(f"  False Negatives : {r['fn']}")
        lines.append(f"  Precision       : {r['precision']:.2f}")
        lines.append(f"  Recall          : {r['recall']:.2f}")
        lines.append(f"  F1              : {r['f1']:.2f}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running evals... (this may take a few minutes)")
    print()

    results = []

    print("[1/4] Injection detection...")
    results.append(eval_injection_detection())

    print("[2/4] PII redaction...")
    results.append(eval_pii_redaction())

    print("[3/4] Toxicity filter...")
    results.append(eval_toxicity_filter())

    print("[4/4] Grounding checker (requires Ollama)...")
    results.append(eval_grounding_checker())

    report = format_report(results)
    print(report)

    report_file = RESULTS / "eval_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to {report_file}")
