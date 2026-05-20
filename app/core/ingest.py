"""Ingesta del corpus documental a ChromaDB.

Uso:
    python -m app.core.ingest --source sample    # solo muestras
    python -m app.core.ingest --source raw       # PDFs completos descargados
    python -m app.core.ingest --source all       # ambas
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from app.core.rag import RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _read_text_file(path: Path) -> str:
    """Lee un archivo de texto plano UTF-8."""
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> List[tuple[int, str]]:
    """Extrae texto por página de un PDF usando pdfplumber.

    Devuelve lista de tuplas (página, texto).
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber no instalado. Ejecute: pip install pdfplumber")
        return []

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def _detect_pais(filename: str) -> str:
    """Detecta jurisdicción a partir del nombre del archivo."""
    low = filename.lower()
    if any(k in low for k in ["dgii", "dominicana", "_do", "rst", "isr"]):
        return "DO"
    return "ES"  # AEAT por defecto


def ingest_directory(directory: Path, rag: RAGPipeline) -> int:
    """Indexa todos los .txt y .pdf de un directorio. Devuelve total de chunks."""
    if not directory.exists():
        logger.warning(f"Directorio no existe: {directory}")
        return 0

    total = 0
    files = sorted(list(directory.glob("*.txt")) + list(directory.glob("*.pdf")))
    if not files:
        logger.warning(f"No hay .txt ni .pdf en {directory}")
        return 0

    for f in files:
        pais = _detect_pais(f.name)
        logger.info(f"Indexando {f.name} (país={pais})...")

        if f.suffix.lower() == ".txt":
            text = _read_text_file(f)
            count = rag.index_text(text, source=f.name, pais=pais, page=1)
            total += count
            logger.info(f"  → {count} chunks")
        elif f.suffix.lower() == ".pdf":
            pages = _read_pdf(f)
            for page_num, text in pages:
                count = rag.index_text(text, source=f.name, pais=pais, page=page_num)
                total += count
            logger.info(f"  → {len(pages)} páginas procesadas")

    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta de corpus documental.")
    parser.add_argument(
        "--source",
        choices=["sample", "raw", "all"],
        default="sample",
        help="Qué corpus indexar: 'sample' (muestras), 'raw' (PDFs oficiales), 'all'.",
    )
    args = parser.parse_args()

    rag = RAGPipeline()

    total = 0
    if args.source in ("sample", "all"):
        logger.info("=" * 60)
        logger.info("INDEXANDO MUESTRAS (data/sample/)")
        logger.info("=" * 60)
        total += ingest_directory(Path("data/sample"), rag)

    if args.source in ("raw", "all"):
        logger.info("=" * 60)
        logger.info("INDEXANDO CORPUS COMPLETO (data/raw/)")
        logger.info("=" * 60)
        total += ingest_directory(Path("data/raw"), rag)

    logger.info("=" * 60)
    logger.info(f"INGESTA COMPLETADA: {total} chunks añadidos")
    logger.info(f"Total documentos en ChromaDB: {rag.count()}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
