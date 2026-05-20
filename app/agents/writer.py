"""Agente Redactor.

Sintetiza la respuesta final integrando el plan y los fragmentos recuperados.

Soporta múltiples ESTRATEGIAS de prompting que el bandit ε-greedy aprende
a elegir según la recompensa observada:
  - "conciso": respuesta breve y directa
  - "detallado": respuesta extensa con razonamiento
  - "paso_a_paso": respuesta estructurada en pasos numerados
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.agents.planner import Plan
from app.core.llm_client import LLMClient
from app.core.rag import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


STRATEGIES = ["conciso", "detallado", "paso_a_paso"]


def _build_system_prompt(strategy: str) -> str:
    """Construye el system prompt según la estrategia elegida."""
    base = (
        "Eres un agente REDACTOR experto en fiscalidad de España (IRPF, autónomos) "
        "y República Dominicana (ISR, RST). Tu respuesta debe basarse EXCLUSIVAMENTE "
        "en los fragmentos de normativa proporcionados. Si la información no es "
        "suficiente, indícalo explícitamente.\n\n"
        "REGLAS OBLIGATORIAS:\n"
        "1. Cita las fuentes al final de cada afirmación importante usando [Fuente N].\n"
        "2. Indica las suposiciones que hagas.\n"
        "3. Finaliza con un aviso legal: este es asesoramiento informativo, no profesional.\n"
        "4. Estructura la respuesta de forma clara con encabezados Markdown.\n"
    )
    if strategy == "conciso":
        return base + "\nESTILO: Respuesta breve (máximo 250 palabras), directa al grano."
    if strategy == "detallado":
        return base + "\nESTILO: Respuesta extensa (400-600 palabras) con razonamiento completo."
    if strategy == "paso_a_paso":
        return base + "\nESTILO: Respuesta estructurada en pasos numerados, ideal para guía práctica."
    return base


@dataclass
class WriterResult:
    """Resultado del agente Redactor."""

    answer: str
    strategy: str
    fuentes_citadas: List[str] = field(default_factory=list)


class WriterAgent:
    """Agente que sintetiza la respuesta final."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def write(
        self,
        question: str,
        plan: Plan,
        chunks: List[Chunk],
        strategy: str = "detallado",
    ) -> WriterResult:
        """Genera la respuesta final.

        Args:
            question: Pregunta original del usuario.
            plan: Plan estructurado del Planificador.
            chunks: Fragmentos recuperados por el Recuperador.
            strategy: Estilo de respuesta ("conciso" | "detallado" | "paso_a_paso").

        Returns:
            WriterResult con la respuesta y las fuentes citadas.
        """
        if strategy not in STRATEGIES:
            logger.warning(f"Estrategia '{strategy}' no reconocida. Uso 'detallado'.")
            strategy = "detallado"

        logger.info(f"[Redactor] Estrategia={strategy}, #chunks={len(chunks)}")

        if not chunks:
            answer = (
                "## ⚠️ Sin información suficiente\n\n"
                "No se han recuperado fragmentos normativos relevantes para responder "
                "esta consulta con la base documental indexada. Recomiendo:\n"
                "1. Reformular la pregunta con términos más específicos.\n"
                "2. Verificar que el corpus para la jurisdicción esté cargado.\n"
                "3. Consultar las fuentes oficiales: AEAT (sede.agenciatributaria.gob.es) "
                "o DGII (dgii.gov.do).\n\n"
                "**Aviso legal:** Este sistema es informativo y no sustituye al "
                "asesoramiento de un profesional habilitado."
            )
            return WriterResult(answer=answer, strategy=strategy, fuentes_citadas=[])

        # Construir el contexto con citas numeradas
        contexto = "\n\n".join(
            f"[Fuente {i+1}] ({c.source}, p.{c.page}, país={c.pais}):\n{c.text}"
            for i, c in enumerate(chunks)
        )
        subtareas_str = "\n".join(f"- {s}" for s in plan.subtareas)

        user_prompt = (
            f"PREGUNTA DEL USUARIO:\n{question}\n\n"
            f"PAÍS DETECTADO: {plan.pais}\n"
            f"PERFIL: {plan.perfil}\n\n"
            f"SUBTAREAS A RESOLVER:\n{subtareas_str}\n\n"
            f"FRAGMENTOS NORMATIVOS DISPONIBLES:\n{contexto}\n\n"
            f"Redacta la respuesta final siguiendo las REGLAS y el ESTILO indicados."
        )

        answer = self.llm.generate(
            prompt=user_prompt,
            system=_build_system_prompt(strategy),
        )

        # Extraer las fuentes citadas (heurística simple por marcadores [Fuente N])
        import re
        fuentes_idx = set(int(m) for m in re.findall(r"\[Fuente\s+(\d+)\]", answer))
        fuentes_citadas = [
            f"{chunks[i-1].source} (p.{chunks[i-1].page})"
            for i in fuentes_idx
            if 1 <= i <= len(chunks)
        ]

        return WriterResult(
            answer=answer,
            strategy=strategy,
            fuentes_citadas=fuentes_citadas,
        )
