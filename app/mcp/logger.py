import os
import sys
from loguru import logger
from loguru._logger import Logger

LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
LOG_LEVEL = os.getenv("CMS_MCP_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True,
)

def get_logger(name: str) -> Logger:
    return logger.bind(name=name)   # type: ignore
