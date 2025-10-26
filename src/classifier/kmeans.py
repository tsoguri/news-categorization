import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.classifier.base import BaseClassifier
from src.classifier.models import HeadlineEntry
from src.embedding.embedding_generator import EmbeddingGenerator
from src.feed.models import FeedContent

logger = logging.getLogger(__name__)


class KMeansClassifier(BaseClassifier):
    """Clusters similar headlines using KMeans on OpenAI text embeddings, with optional visualization saving."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        max_workers: int = 8,
        n_clusters: Optional[int] = None,
        visualize_plots: bool = True,
        visualization_dir: str = "visualizations/kmeans",
    ):
        self.embedding_generator = EmbeddingGenerator(
            model=embedding_model, max_workers=max_workers
        )
        self.n_clusters = n_clusters
        self.optimize = n_clusters is None
        self.visualize_plots = visualize_plots
        self.visualization_dir = visualization_dir

        os.makedirs(self.visualization_dir, exist_ok=True)
        logger.info(
            f"Initialized KMeansClassifier with n_clusters={n_clusters}, model={embedding_model}, "
            f"visualizations_dir={self.visualization_dir}"
        )

    def _save_visualization(
        self, tested_ks, inertias, silhouette_scores, davies_bouldin_scores
    ):
        """Save elbow, silhouette, and DB index plots to disk instead of showing them."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Elbow plot
        axes[0].plot(tested_ks, inertias, "bo-", linewidth=2, markersize=4)
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Elbow Method")
        axes[0].grid(True, alpha=0.3)

        # Silhouette plot
        best_k_sil = tested_ks[np.argmax(silhouette_scores)]
        axes[1].plot(tested_ks, silhouette_scores, "go-", linewidth=2, markersize=4)
        axes[1].axvline(
            best_k_sil, color="red", linestyle="--", label=f"Best k={best_k_sil}"
        )
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Score (Higher = Better)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Davies-Bouldin plot
        best_k_db = tested_ks[np.argmin(davies_bouldin_scores)]
        axes[2].plot(tested_ks, davies_bouldin_scores, "ro-", linewidth=2, markersize=4)
        axes[2].axvline(
            best_k_db, color="red", linestyle="--", label=f"Best k={best_k_db}"
        )
        axes[2].set_xlabel("Number of Clusters (k)")
        axes[2].set_ylabel("Davies-Bouldin Score")
        axes[2].set_title("Davies-Bouldin Score (Lower = Better)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()

        filename = os.path.join(self.visualization_dir, "kmeans_metrics.png")
        plt.savefig(filename)
        plt.close(fig)

        logger.info(f"Saved KMeans metric visualizations → {filename}")

    def _find_optimal_clusters(self) -> int:
        """Find optimal number of clusters using elbow, silhouette, and DB index."""
        max_possible_k = len(self.embeddings) - 1
        k_range = range(2, max(3, max_possible_k + 1))

        inertias, silhouette_scores, davies_bouldin_scores, tested_ks = [], [], [], []

        logger.info("Finding optimal number of clusters...")

        for k in k_range:
            if (100 < k < 500 and k % 10 != 0) or (k >= 500 and k % 100 != 0):
                continue

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.embeddings, labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.embeddings, labels))
            tested_ks.append(k)

            if k % 10 == 0 or k < 10:
                logger.info(f"k={k}: silhouette={silhouette_scores[-1]:.3f}")

        if self.visualize_plots:
            self._save_visualization(
                tested_ks, inertias, silhouette_scores, davies_bouldin_scores
            )

        optimal_k = tested_ks[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters (by silhouette): {optimal_k}")
        return optimal_k

    def classify(self, content_feeds: list[FeedContent]) -> list[HeadlineEntry]:
        """Cluster feed entries into similar headlines using KMeans."""
        headlines = {
            f"{e.title} {e.detail}": e for cf in content_feeds for e in cf.entries
        }
        if not headlines:
            logger.warning("No feed entries provided for classification.")
            return []

        texts = list(headlines.keys())
        self.embeddings = self.embedding_generator.generate_embeddings(texts)

        if self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters()

        logger.info(f"Clustering into {self.n_clusters} groups...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)

        headline_entries = []
        for cluster_id in range(self.n_clusters):
            cluster_texts = [
                t for t, label in zip(texts, labels) if label == cluster_id
            ]
            if not cluster_texts:
                continue
            cluster_entry = HeadlineEntry(
                headline_text=headlines[cluster_texts[0]].title,
                entries=[headlines[t] for t in cluster_texts],
            )
            headline_entries.append(cluster_entry)

        logger.info(f"Generated {len(headline_entries)} headline clusters.")
        self.headline_entries = headline_entries
        return headline_entries

    def visualize_clusters(self, return_df=False):
        if self.optimize:
            self._show_saved_visualizations()
        return super().visualize_clusters(return_df=return_df)

    def _show_saved_visualizations(self):
        """Show previously saved KMeans visualizations, if available."""
        if not os.path.exists(self.visualization_dir):
            logger.warning(f"No visualization folder found: {self.visualization_dir}")
            return

        files = [
            f
            for f in os.listdir(self.visualization_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            logger.warning(f"No visualization images found in {self.visualization_dir}")
            return

        for f in files:
            filepath = os.path.join(self.visualization_dir, f)
            logger.info(f"Showing visualization: {filepath}")
            img = plt.imread(filepath)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f)
            plt.show()
