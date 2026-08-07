"""
Centralized logging utility for the ARQPO project.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name, usually __name__.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
