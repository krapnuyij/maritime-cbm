"""Application logging configuration."""

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure standard-library logging for command-line entry points."""
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT, force=True)
