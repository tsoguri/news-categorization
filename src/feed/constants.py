from src.feed.base import Feed
from src.feed.google import GoogleFeed

COMMON_FEEDS: list[Feed] = [
    Feed(
        publication="NY Times",
        url="https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    ),
    Feed(
        publication="WSJ",
        category="World",
        url="https://feeds.content.dowjones.io/public/rss/RSSWorldNews",
    ),
    Feed(
        publication="Fox News",
        url="https://moxie.foxnews.com/google-publisher/latest.xml",
    ),
    Feed(publication="CNN", url="http://rss.cnn.com/rss/cnn_topstories.rss"),
    Feed(publication="NPR", url="https://feeds.npr.org/1001/rss.xml"),
    GoogleFeed(
        publication="Google News",
        url="https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    ),
]
