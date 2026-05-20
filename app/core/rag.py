"""Pipeline RAG (Retrieval-Augmented Generation).

Funciones:
- Indexar documentos PDF y de texto en ChromaDB con metadatos de jurisdicción.
- Recuperar fragmentos relevantes mediante embeddings multilingües.

Decisiones de diseño:
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` por su balance entre
  rendimiento y tamaño, y por soportar español multilingüe (ES y DO).
- Chunking: 600 tokens con solapamiento de 80 para preservar contexto entre
  cortes de párrafo. Estos valores son típicos para normativa densa.
- Vector store: ChromaDB persistente en disco — sin servidor, ideal para
  Render free tier y portabilidad local.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Fragmento de texto con metadatos."""

    text: str
    source: str
    pais: str  # "ES" | "DO"
    page: Optional[int] = None
    chunk_id: str = ""


@dataclass
class RAGConfig:
    """Configuración del pipeline RAG."""

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    persist_dir: str = "./chroma_db"
    collection_name: str = "normativa_fiscal"
    chunk_size: int = 600
    chunk_overlap: int = 80
    top_k: int = 4


class RAGPipeline:
    """Pipeline RAG completo: indexación + recuperación."""

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig(
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            top_k=int(os.getenv("MAX_RETRIEVAL_DOCS", "4")),
        )
        Path(self.config.persist_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Cargando modelo de embeddings: {self.config.embedding_model}")
        self.embedder = SentenceTransformer(self.config.embedding_model)

        self.client = chromadb.PersistentClient(
            path=self.config.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB lista. Documentos en colección: {self.collection.count()}"
        )

    # -------- Chunking --------

    def _split_text(self, text: str) -> List[str]:
        """Divide texto en chunks con solapamiento.

        Estrategia simple por longitud de caracteres, suficiente para
        normativa fiscal donde los párrafos son moderadamente uniformes.
        """
        # Aproximación: 1 token ≈ 4 caracteres en español
        chunk_chars = self.config.chunk_size * 4
        overlap_chars = self.config.chunk_overlap * 4

        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= chunk_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_chars
            # Cortar preferiblemente en un punto cercano para no partir frases
            if end < len(text):
                last_period = text.rfind(". ", start, end)
                if last_period > start + chunk_chars // 2:
                    end = last_period + 1
            chunks.append(text[start:end].strip())
            start = end - overlap_chars if end < len(text) else end
        return [c for c in chunks if len(c) > 50]  # descartar fragmentos muy cortos

    # -------- Indexación --------

    def index_text(
        self,
        text: str,
        source: str,
        pais: str,
        page: Optional[int] = None,
    ) -> int:
        """Indexa un bloque de texto en la colección.

        Args:
            text: Contenido textual.
            source: Nombre del documento fuente (ej. "ManualRenta2024Tomo1.pdf").
            pais: Jurisdicción ("ES" o "DO").
            page: Número de página de origen, si aplica.

        Returns:
            Número de chunks indexados.
        """
        chunks = self._split_text(text)
        if not chunks:
            return 0

        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{source}_{page or 0}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source, "pais": pais, "page": page or 0}
            for _ in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    # -------- Recuperación --------

    def retrieve(
        self,
        query: str,
        pais: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Chunk]:
        """Recupera los fragmentos más relevantes para una consulta.

        Args:
            query: Texto de búsqueda.
            pais: Filtro por jurisdicción ("ES" o "DO"). None = todas.
            top_k: Número de resultados. Por defecto config.top_k.

        Returns:
            Lista de Chunks ordenados por relevancia descendente.
        """
        k = top_k or self.config.top_k
        if self.collection.count() == 0:
            logger.warning("ChromaDB vacía: no hay documentos indexados aún.")
            return []

        query_emb = self.embedder.encode([query]).tolist()
        where_filter = {"pais": pais} if pais else None

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
            where=where_filter,
        )

        chunks = []
        if not results["documents"] or not results["documents"][0]:
            return chunks

        for doc, meta, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
        ):
            chunks.append(
                Chunk(
                    text=doc,
                    source=meta.get("source", "desconocido"),
                    pais=meta.get("pais", "ES"),
                    page=meta.get("page"),
                    chunk_id=cid,
                )
            )
        return chunks

    def count(self) -> int:
        """Número de documentos indexados."""
        return self.collection.count()
