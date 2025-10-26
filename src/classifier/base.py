from abc import ABC, abstractmethod
from typing import Optional

from src.classifier.models import HeadlineEntry
from src.feed.models import FeedContent

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


class BaseClassifier(ABC):
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.headline_entries: Optional[list[HeadlineEntry]] = None

    @abstractmethod
    def classify(self, content_feeds: list[FeedContent]) -> list[HeadlineEntry]:
        pass

    def visualize_clusters(self, return_df: bool = False):
        """
        Visualize clustered headline embeddings in 2D space using PCA.
        Each HeadlineEntry represents a cluster.
        """
        if not self.headline_entries:
            raise ValueError(
                "self.headline_entries not found. Did you classify headlines?"
            )

        if self.embeddings is None:
            raise ValueError("self.embeddings not found. Did you compute embeddings?")

        total_entries = sum(len(h.entries) for h in self.headline_entries)
        if len(self.embeddings) != total_entries:
            raise ValueError(
                f"Embeddings count ({len(self.embeddings)}) does not match total FeedEntry count ({total_entries})."
            )

        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(self.embeddings)

        clusters = []
        headlines = []

        for cluster_idx, headline_entry in enumerate(self.headline_entries):
            for feed_entry in headline_entry.entries:
                clusters.append(cluster_idx)
                headlines.append(
                    feed_entry.title[:10] or headline_entry.headline_text[:10]
                )

        df = pd.DataFrame(
            {
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "cluster": clusters,
                "headline": headlines,
            }
        )

        n_clusters = len(self.headline_entries)
        plt.figure(figsize=(12, 6))
        colors = sns.color_palette("husl", n_clusters)

        for i in range(n_clusters):
            cluster_data = df[df["cluster"] == i]
            plt.scatter(
                cluster_data["x"],
                cluster_data["y"],
                c=[colors[i]],
                label=f"Cluster {i}",
                s=200,
                alpha=0.6,
                edgecolors="black",
                linewidth=1.5,
            )

            for _, row in cluster_data.iterrows():
                plt.annotate(
                    row["headline"][:40] + "...",
                    (row["x"], row["y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

        plt.title(
            "News Headlines Clustered by Topic\n(using OpenAI Embeddings)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel(
            f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            fontsize=12,
        )
        plt.ylabel(
            f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            fontsize=12,
        )
        plt.legend(fontsize=10, loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        if return_df:
            return df
