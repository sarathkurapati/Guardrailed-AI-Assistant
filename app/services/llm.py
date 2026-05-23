from langchain_ollama import OllamaLLM
from app.config import settings

_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
            num_predict=512,
        )
    return _llm
