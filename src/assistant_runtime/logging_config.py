"""Logging configuration — Loguru setup with stdlib intercept handler."""

import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route stdlib logging through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(json_output: bool = False, level: str = "INFO") -> None:
    """Configure Loguru as the sole logging handler.

    Args:
        json_output: If True, emit structured JSON logs. False for human-readable.
        level: Minimum log level.
    """
    logger.remove()

    if json_output:
        logger.add(sys.stdout, level=level, serialize=True, enqueue=True)
    else:
        logger.add(
            sys.stdout,
            level=level,
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
            ),
            enqueue=True,
        )

    # Intercept stdlib logging (uvicorn, fastapi, httpx)
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party loggers
    for noisy_logger in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
