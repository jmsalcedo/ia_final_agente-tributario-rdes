"""Logger común del proyecto con formato uniforme."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Devuelve un logger configurado con formato uniforme.

    Args:
        name: Nombre del logger (típicamente __name__).
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)

    # Evitar añadir handlers duplicados si el logger ya está configurado
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Handler a consola
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Handler a archivo (opcional, si existe carpeta logs)
    logs_dir = Path("logs")
    if logs_dir.exists():
        file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
