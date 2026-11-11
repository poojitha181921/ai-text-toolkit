import logging
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "toolkit.log"

def setup_logger(name: str = "ai_toolkit") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        # Stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_fmt = logging.Formatter("[%(levelname)s] %(message)s")
        stream_handler.setFormatter(stream_fmt)
        logger.addHandler(stream_handler)

    return logger