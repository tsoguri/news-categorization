import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import numpy as np

from src.client import OPENAI_CLIENT

logger = logging.getLogger(__name__)

# ---- Shared cache across all EmbeddingGenerator instances ----
_GLOBAL_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}


class EmbeddingGenerator:
    """Handles generation of embeddings using OpenAI's API (concurrently, with global caching)."""

    def __init__(self, model: str = "text-embedding-3-small", max_workers: int = 8):
        self.model = model
        self.max_workers = max_workers
        logger.info(f"Initialized EmbeddingGenerator with model: {model}")

    def _generate_single(self, text: str, idx: int) -> np.ndarray:
        """Helper function to generate a single embedding."""
        logger.debug(f"Generating embedding for text index {idx}")
        response = OPENAI_CLIENT.embeddings.create(input=text, model=self.model)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        _GLOBAL_EMBEDDING_CACHE[text] = embedding
        return embedding

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts concurrently, using a global cache."""
        if not texts:
            raise ValueError("Text list cannot be empty")

        embeddings = [None] * len(texts)
        uncached = [
            (i, t) for i, t in enumerate(texts) if t not in _GLOBAL_EMBEDDING_CACHE
        ]
        cached = [(i, t) for i, t in enumerate(texts) if t in _GLOBAL_EMBEDDING_CACHE]

        logger.info(
            f"{len(cached)} cached, {len(uncached)} new to fetch (model={self.model})."
        )

        # Fill cached immediately
        for i, t in cached:
            embeddings[i] = _GLOBAL_EMBEDDING_CACHE[t]

        # Generate new ones concurrently
        if uncached:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._generate_single, t, i): i for i, t in uncached
                }
                for count, fut in enumerate(as_completed(futures), 1):
                    i = futures[fut]
                    try:
                        embeddings[i] = fut.result()
                        if count % 10 == 0 or count == len(uncached):
                            logger.debug(f"Processed {count}/{len(uncached)} new texts")
                    except Exception as e:
                        logger.error(f"Error embedding index {i}: {e}")
                        raise

        result = np.vstack(embeddings)
        logger.info(
            f"Embeddings generated with shape {result.shape}. Cache size: {len(_GLOBAL_EMBEDDING_CACHE)}"
        )
        return result
