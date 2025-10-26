from pydantic import BaseModel

from src.feed.models import FeedEntry


class Headline(BaseModel):
    headline_text: str


class HeadlineEntry(BaseModel):
    headline_text: str
    entries: list[FeedEntry]


class HeadlineEntries:
    def __init__(self):
        self.entries: list[HeadlineEntry] = []

    def add(self, headline_text: str, entry: FeedEntry):
        for headline_entry in self.entries:
            if headline_entry.headline_text == headline_text:
                headline_entry.entries.append(entry)
                return
        self.entries.append(HeadlineEntry(headline_text=headline_text, entries=[entry]))

    def get_headlines(self) -> list[str]:
        return [he.headline_text for he in self.entries]
