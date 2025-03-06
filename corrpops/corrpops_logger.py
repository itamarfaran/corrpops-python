import logging

logging.basicConfig(level=logging.INFO)


def corrpops_logger() -> logging.Logger:
    return logging.getLogger("corrpops")
