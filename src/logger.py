"""Project-wide logging: every module calls get_logger() and logs to console and logs/project.log."""
import logging
import sys

from .config import LOGS_DIR

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "rossmann", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (handlers are added once, so re-running a cell never duplicates lines)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logfile = logging.FileHandler(LOGS_DIR / "project.log", encoding="utf-8")
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)
    return logger
