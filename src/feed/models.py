from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class FeedMedia(BaseModel):
    medium: Optional[Literal["image"]] = "image"
    url: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None


class FeedEntry(BaseModel):
    title: Optional[str] = None
    detail: Optional[str] = None
    subtitle: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    link: Optional[str] = None
    published_datetime: Optional[datetime] = None
    tags: Optional[list[str]] = None
    media: Optional[list[FeedMedia]] = None


class FeedContent(BaseModel):
    feed_publication: str
    feed_category: str
    feed_name: str
    feed_url: str
    feed_frequency: str
    title: Optional[str] = None
    link: Optional[str] = None
    description: Optional[str] = None
    image: Optional[FeedMedia] = None
    updated_datetime: Optional[datetime] = None
    entries: Optional[list[FeedEntry]] = None
