from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from app.utils.logging import get_logger

logger = get_logger(__name__)

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None

ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "NRP",
    "URL",
    # DATE_TIME excluded: produces high false-positive rate on policy text
    # (e.g. "20 days", "5 working days" incorrectly classified as personal dates).
    # In a real deployment, a custom recognizer with date-format pattern matching
    # would be used instead of the generic DATE_TIME recogniser.
]


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
        logger.info("presidio_analyzer_loaded")
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
        logger.info("presidio_anonymizer_loaded")
    return _anonymizer


def analyze_text(text: str) -> list:
    analyzer = get_analyzer()
    # score_threshold=0.7 reduces false positives on policy text
    # (e.g. "20 days", "5 working days" incorrectly flagged as DATE_TIME at lower thresholds)
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=ENTITIES,
        score_threshold=0.5,
    )
    return results


def anonymize_text(text: str, analyzer_results: list) -> str:
    if not analyzer_results:
        return text
    anonymizer = get_anonymizer()
    operators = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in ENTITIES
    }
    result = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators,
    )
    return result.text
