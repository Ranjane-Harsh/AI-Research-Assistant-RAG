from langchain_huggingface import HuggingFaceEmbeddings
import config

_EMBEDDING_MODEL = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the HuggingFace sentence-transformer model in memory.
    Subsequent calls reuse the cached singleton.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDING_MODEL
