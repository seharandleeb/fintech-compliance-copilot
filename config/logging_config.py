"""
Application logging configuration.

This module configures Loguru for the entire application.

Every module should import the logger from here instead of
creating its own logger instance.

Example:
    from config.logging_config import logger

    logger.info("Loading dataset...")
"""

import sys

from loguru import logger

from config.settings import get_settings

settings = get_settings()

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging() -> None:
    """
    Configure application logging.

    Two logging sinks are created:

    1. Console logging
    2. Rotating log file

    This function should be called only once during
    application startup.
    """

    # Remove Loguru default logger
    logger.remove()

    # Console logger
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=LOG_FORMAT,
        colorize=True,
    )

    # File logger
    logger.add(
        settings.logs_dir / "application.log",
        level=settings.log_level,
        format=LOG_FORMAT,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
    )


# Configure logging automatically when imported
configure_logging()