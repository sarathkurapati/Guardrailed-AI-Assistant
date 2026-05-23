from transformers import pipeline as hf_pipeline
from app.utils.logging import get_logger

logger = get_logger(__name__)

_toxicity_pipeline = None


def get_toxicity_pipeline():
    global _toxicity_pipeline
    if _toxicity_pipeline is None:
        _toxicity_pipeline = hf_pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None,
        )
        logger.info("toxic_bert_loaded")
    return _toxicity_pipeline


def score_toxicity(text: str) -> float:
    if not text or len(text.strip()) < 3:
        return 0.0
    pipe = get_toxicity_pipeline()
    try:
        results = pipe(text[:512])  # truncate to model max
        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]
        for item in results:
            if item["label"].lower() == "toxic":
                return float(item["score"])
        return 0.0
    except Exception as e:
        logger.warning("toxicity_score_error", error=str(e))
        return 0.0
