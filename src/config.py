import logging
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    """Configure logging for the categorization module.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_str: Custom format string for log messages.
        filename: Optional filename to write logs to.
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_str,
        filename=filename,
        force=True,  # Override any existing configuration
    )

    # Set specific logger levels for our modules
    logging.getLogger("src.categorization").setLevel(numeric_level)

    # Reduce noise from external libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
