from abc import ABC
from datetime import datetime
from time import mktime
from typing import Literal, Optional

import feedparser

from src.feed.models import FeedContent, FeedEntry, FeedMedia


class Feed(ABC):
    def __init__(
        self,
        publication: str,
        url: str,
        category: Optional[
            Literal[
                "Top Stories",
                "World",
                "NY",
                "US",
                "Politics",
                "Technology",
                "Science",
                "Business",
                "Economy",
            ]
        ] = "Top Stories",
        frequency: Optional[Literal["hourly", "daily", "weekly", "monthly"]] = "hourly",
    ):
        self.publication = publication
        self.category = category
        self.name = f"{publication} - {category}"
        self.url = url
        self.frequency = frequency

    detail: Optional[str] = None

    def parse_feed(self) -> FeedContent:
        parsed = feedparser.parse(self.url)
        parsed_feed = parsed.feed
        parsed_entries = parsed.entries

        entries = []
        for entry in parsed_entries:
            published_parsed = getattr(entry, "published_parsed", None)
            tags_parsed = getattr(entry, "tags", [])
            tags = [t.get("term") for t in tags_parsed]
            media_parsed = getattr(entry, "media_content", [])
            media = [
                FeedMedia(
                    url=m.get("url"),
                    medium=m.get("medium"),
                    height=m.get("height"),
                    width=m.get("width"),
                )
                for m in media_parsed
            ]

            entries.append(
                FeedEntry(
                    title=getattr(entry, "title", None),
                    detail=self._return_detail(
                        subtitle=getattr(entry, "subtitle", None),
                        summary=getattr(entry, "summary", None),
                        description=getattr(entry, "description", None),
                    ),
                    subtitle=getattr(entry, "subtitle", None),
                    summary=getattr(entry, "summary", None),
                    description=getattr(entry, "description", None),
                    author=getattr(entry, "author", None),
                    link=getattr(entry, "link", None),
                    published_datetime=(
                        datetime.fromtimestamp(mktime(published_parsed))
                        if published_parsed
                        else None
                    ),
                    tags=tags if tags else None,
                    media=media if media else None,
                )
            )

        image_href = getattr(getattr(parsed_feed, "image", None), "href", None)
        updated_parsed = getattr(parsed_feed, "updated_parsed", None)

        return FeedContent(
            feed_publication=self.publication,
            feed_category=self.category,
            feed_name=self.name,
            feed_url=self.url,
            feed_frequency=self.frequency,
            title=getattr(parsed_feed, "title", None),
            link=getattr(parsed_feed, "link", None),
            description=getattr(parsed_feed, "description", None),
            image=FeedMedia(url=image_href) if image_href else None,
            updated_datetime=(
                datetime.fromtimestamp(mktime(updated_parsed))
                if updated_parsed
                else None
            ),
            entries=entries,
        )

    def _return_detail(
        self,
        subtitle: Optional[str],
        summary: Optional[str],
        description: Optional[str],
    ):
        parts = []
        if subtitle:
            parts.append(subtitle)

        if summary and summary != description:
            parts.append(summary)
        elif summary:
            parts.append(description)

        result = " ".join(parts)
        return result
