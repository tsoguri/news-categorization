import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.classifier.base import BaseClassifier
from src.classifier.models import Headline, HeadlineEntries, HeadlineEntry
from src.client import OPENAI_CLIENT
from src.feed.models import FeedContent, FeedEntry

logger = logging.getLogger(__name__)


class LLMClassifier(BaseClassifier):
    def __init__(self, model: str = "gpt-5-mini", max_workers: int = 8):
        self.model = model
        self.headline_entries = HeadlineEntries()
        self.max_workers = max_workers

    def _classify_entry(
        self, cf_i: int, cf: FeedContent, entry_i: int, entry: FeedEntry
    ) -> tuple[str, FeedEntry]:
        """
        Helper function to classify a single feed entry.
        """
        logger.info(
            f"LLM classifying feed entry {entry_i + 1} / {len(cf.entries)} for feed `{cf.feed_name}` {cf_i + 1}"
        )

        # Prepare prompt
        prompt = f"""
        You are a news headline classifier. 

        Existing headline buckets:
        {self.headline_entries.get_headlines()}

        Classify the following article title into one of the existing headline buckets if it is similar in meaning,
        or create a new headline if it does not match any existing ones.

        Article title:
        {entry.detail}
        """

        response = OPENAI_CLIENT.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            text_format=Headline,
        )
        headline_text = response.output_parsed.headline_text
        return headline_text, entry

    def classify(self, content_feeds: list[FeedContent]) -> list[HeadlineEntry]:
        if not content_feeds:
            return []
        first_feed = content_feeds[0]
        first_entry = first_feed.entries[0]
        self.headline_entries.add(headline_text=first_entry.detail, entry=first_entry)

        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for cf_i, cf in enumerate(content_feeds):
                for entry_i, entry in enumerate(cf.entries):
                    if cf_i == 0 and entry_i == 0:
                        continue
                    tasks.append(
                        executor.submit(self._classify_entry, cf_i, cf, entry_i, entry)
                    )
            for future in as_completed(tasks):
                try:
                    headline_text, entry = future.result()
                    self.headline_entries.add(headline_text=headline_text, entry=entry)
                except Exception as e:
                    logger.error(f"Error classifying entry: {e}")

        return self.headline_entries.entries
