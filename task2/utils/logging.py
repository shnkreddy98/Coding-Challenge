import logging
import os

from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def custom_logging():
    log_file = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%y%m%dT%H%M%S')}.log")
    FMT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=FMT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
