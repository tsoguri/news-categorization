import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from src.classifier.base import BaseClassifier
from src.classifier.models import HeadlineEntry
from src.embedding.embedding_generator import EmbeddingGenerator
from src.feed.models import FeedContent

logger = logging.getLogger(__name__)


class DBScanClassifier(BaseClassifier):
    """Clusters similar headlines using DBSCAN on OpenAI text embeddings."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        max_workers: int = 8,
        eps: float = 0.19,
        min_samples: int = 2,
    ):
        self.embedding_generator = EmbeddingGenerator(
            model=embedding_model, max_workers=max_workers
        )
        self.eps = eps
        self.min_samples = min_samples
        logger.info(
            f"Initialized DBScanClassifier with eps={eps}, min_samples={min_samples}, model={embedding_model}"
        )

    def classify(self, content_feeds: list[FeedContent]) -> list[HeadlineEntry]:
        """Cluster feed entries into similar headlines using DBSCAN."""
        headlines = {
            f"{e.title} {e.detail}": e for cf in content_feeds for e in cf.entries
        }

        if not headlines:
            logger.warning("No feed entries provided for classification.")
            return []

        texts = list(headlines.keys())
        logger.info(f"Generating embeddings for {len(texts)} headlines...")
        self.embeddings = self.embedding_generator.generate_embeddings(texts)
        embeddings_normalized = normalize(self.embeddings)

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        cluster_labels = dbscan.fit_predict(embeddings_normalized)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)

        logger.info(f"DBSCAN found {n_clusters} clusters.")
        logger.info(f"Noise points (unclustered): {n_noise}")

        if n_noise > 0:
            logger.info(
                f"{n_noise} headlines are unique enough to stand alone. "
                f"Consider increasing eps (currently {self.eps}) to cluster more tightly related items."
            )

        headline_entries: list[HeadlineEntry] = []

        for cluster_id in range(n_clusters):
            cluster_texts = [
                text
                for text, label in zip(texts, cluster_labels)
                if label == cluster_id
            ]
            if not cluster_texts:
                continue

            cluster_title = headlines[cluster_texts[0]].title
            cluster_entry = HeadlineEntry(
                headline_text=cluster_title,
                entries=[headlines[t] for t in cluster_texts],
            )
            headline_entries.append(cluster_entry)

        noise_texts = [
            text for text, label in zip(texts, cluster_labels) if label == -1
        ]
        for text in noise_texts:
            entry_obj = headlines[text]
            noise_entry = HeadlineEntry(
                headline_text=entry_obj.title, entries=[entry_obj]
            )
            headline_entries.append(noise_entry)

        logger.info(f"Generated {len(headline_entries)} total HeadlineEntry objects.")
        self.headline_entries = headline_entries
        return headline_entries
