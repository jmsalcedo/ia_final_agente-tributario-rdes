"""Agente Recuperador.

Su responsabilidad es ejecutar el pipeline RAG: por cada subtarea del plan,
buscar los fragmentos más relevantes en ChromaDB, filtrando por jurisdicción.

Devuelve una lista deduplicada de chunks con metadatos.
"""
from __future__ import annotations

from typing import List, Optional

from app.agents.planner import Plan
from app.core.rag import Chunk, RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverAgent:
    """Agente que ejecuta consultas RAG según el plan."""

    def __init__(self, rag: Optional[RAGPipeline] = None) -> None:
        self.rag = rag or RAGPipeline()

    def retrieve_for_plan(
        self, plan: Plan, query_original: str, top_k_per_query: int = 3
    ) -> List[Chunk]:
        """Recupera fragmentos para cada subtarea del plan.

        Args:
            plan: Plan generado por el Planificador.
            query_original: Pregunta original del usuario (refuerza la consulta).
            top_k_per_query: Cuántos chunks por subtarea.

        Returns:
            Lista deduplicada de chunks, ordenada por aparición.
        """
        logger.info(
            f"[Recuperador] Ejecutando {len(plan.subtareas)} consultas "
            f"para país={plan.pais}"
        )
        seen_ids: set[str] = set()
        all_chunks: List[Chunk] = []

        # Consulta inicial con la pregunta original
        chunks = self.rag.retrieve(query_original, pais=plan.pais, top_k=top_k_per_query)
        for c in chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                all_chunks.append(c)

        # Consulta por cada subtarea
        for subtask in plan.subtareas:
            chunks = self.rag.retrieve(subtask, pais=plan.pais, top_k=top_k_per_query)
            for c in chunks:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    all_chunks.append(c)

        logger.info(
            f"[Recuperador] {len(all_chunks)} fragmentos únicos recuperados"
        )
        return all_chunks
