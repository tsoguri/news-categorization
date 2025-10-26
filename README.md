# News Categorization

A Python project inspired by my love of Google News — I read it every day and have always been curious about how it groups articles into relevant headlines. I wanted to see what a simple approach might look like, so in this project I experiment with automatically categorizing news headlines from multiple RSS feeds using straightforward machine learning and AI techniques.

## Overview

This project fetches news headlines from RSS feeds and categorizes them using different clustering and classification techniques. The goal is to group similar news stories together and compare the effectiveness of various approaches.

## Features

- **RSS Feed Processing**: Fetches and parses news from multiple sources
- **Multiple Classification Approaches**:
  - **K-Means Clustering**: Groups headlines using vector embeddings
  - **DBSCAN Clustering**: Density-based clustering with noise detection
  - **LLM Classification**: AI-powered semantic categorization
- **Embedding Generation**: Uses OpenAI's text-embedding-3-small model
- **Visualization**: Generates cluster visualizations for analysis

## Classification Methods Implemented

### K-Means Clustering
- Fixed cluster count (configurable)
- Automatic optimization using silhouette score
- Generates visualizations of cluster distributions

### DBSCAN Clustering
- Density-based clustering with configurable epsilon values
- Identifies noise points (standalone headlines)
- Multiple configurations tested (eps: 0.18, 0.35, 0.5, 0.65)

### LLM Classification
- Uses OpenAI's LLMs for label classification
- Provides more contextually aware categorization

## Installation

Requires Python 3.12+. Install dependencies:

```bash
uv sync
```

## Usage

The main analysis is contained in `classification_analysis.ipynb`. This notebook:

1. Fetches news from all configured RSS feeds
2. Deduplicates headlines across sources
3. Runs multiple classification approaches
4. Compares results across different methods

### Running Classifications

```python
from src.feed.common import deduplicate_feeds
from src.feed.constants import COMMON_FEEDS
from src.classifier.kmeans import KMeansClassifier
from src.classifier.dbscan import DBScanClassifier
from src.classifier.llm import LLMClassifier

# Fetch and deduplicate feeds
feeds = COMMON_FEEDS
content_feeds = [feed.parse_feed() for feed in feeds]
unique_feeds = deduplicate_feeds(content_feeds)

# Run different classifiers
kmeans_classifier = KMeansClassifier(n_clusters=100)
dbscan_classifier = DBScanClassifier(eps=0.35)
llm_classifier = LLMClassifier()

# Classify headlines
kmeans_results = kmeans_classifier.classify(unique_feeds)
dbscan_results = dbscan_classifier.classify(unique_feeds)
llm_results = llm_classifier.classify(unique_feeds)
```

## Configuration

- **News Sources**: Configured in `src/feed/constants.py`
- **Clustering Parameters**: Adjustable in classifier constructors
- **Visualizations**: Saved to `visualizations/` directory

## Analysis Results

The project compares different approaches across metrics like:
- Number of clusters/categories generated
- Average cluster size
- Distribution of cluster sizes
- Noise detection (for DBSCAN)
- Semantic coherence of groupings

Example comparison shows:
- K-Means (70 clusters): Even distribution, fixed grouping
- DBSCAN (eps=0.35): 42 clusters + 213 standalone headlines
- LLM Classification: Context-aware semantic grouping
