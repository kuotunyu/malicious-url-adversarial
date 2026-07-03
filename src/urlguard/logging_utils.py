"""統一的 logger 工廠,取代 notebook 裡散落的 print()。"""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "urlguard", level: int = logging.INFO) -> logging.Logger:
    """回傳設定好格式的 logger(整個程序只設定一次 root handler)。"""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
