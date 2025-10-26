from datetime import datetime
from typing import Literal, Optional

import feedparser
from bs4 import BeautifulSoup

from src.feed.base import Feed
from src.feed.models import FeedContent, FeedEntry


class GoogleFeed(Feed):
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

        entries: list[FeedEntry] = []

        for cluster in parsed_entries:
            cluster_title = cluster.get("title", "").strip()
            cluster_date = None
            if cluster.get("published_parsed"):
                cluster_date = datetime(*cluster.published_parsed[:6])

            desc_html = cluster.get("description", "")
            soup = BeautifulSoup(desc_html, "html.parser")

            for li in soup.find_all("li"):
                a = li.find("a")
                if not a:
                    continue
                link = a.get("href")
                title = a.get_text(strip=True)
                font_tag = li.find("font")
                source = font_tag.get_text(strip=True) if font_tag else None

                entry = FeedEntry(
                    title=title,
                    author=source,
                    link=link,
                    published_datetime=cluster_date,
                    tags=[cluster_title],
                    description=cluster_title,
                )
                entries.append(entry)

        return FeedContent(
            feed_publication=self.publication,
            feed_category=self.category,
            feed_name=self.name,
            feed_url=self.url,
            feed_frequency=self.frequency,
            title=getattr(parsed_feed, "title", None),
            link=getattr(parsed_feed, "link", None),
            description=getattr(parsed_feed, "description", None),
            updated_datetime=datetime.utcnow(),
            entries=entries,
        )
