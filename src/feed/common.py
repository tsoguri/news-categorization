from src.feed.models import FeedContent


def deduplicate_feeds(content_feeds: list[FeedContent]):
    seen_entries = set()
    for cf in content_feeds:
        unique_entries = [
            entry for entry in cf.entries if entry.link not in seen_entries
        ]
        seen_entries.update(entry.link for entry in unique_entries)
        cf.entries = unique_entries
    return content_feeds
