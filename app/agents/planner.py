"""Agente Planificador.

Su responsabilidad es analizar la pregunta del usuario y producir un plan
de subtareas, además de detectar la jurisdicción aplicable (ES o DO).

Salida: estructura JSON con `pais`, `subtareas` y `perfil` detectado.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.llm_client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT_PLANIFICADOR = """Eres un agente PLANIFICADOR especializado en fiscalidad.
Tu única tarea es analizar la pregunta del usuario y producir un plan estructurado en JSON.

Detecta:
1. País aplicable: "ES" (España) o "DO" (República Dominicana). Si la pregunta menciona
   "España", "AEAT", "IRPF", "autónomo español" → ES. Si menciona "República Dominicana",
   "DGII", "ISR", "RST" → DO. Si es ambiguo, usa "ES" por defecto y márcalo en el plan.
2. Subtareas concretas (3 a 5) ordenadas para resolver la pregunta.
3. Perfil del contribuyente si se infiere (autónomo, asalariado, sociedad, etc.).

Responde ÚNICAMENTE con un objeto JSON con esta estructura exacta:
{
  "pais": "ES" | "DO",
  "perfil": "string breve",
  "subtareas": ["subtarea 1", "subtarea 2", ...]
}
No incluyas explicaciones, sólo el JSON."""


@dataclass
class Plan:
    """Plan generado por el Planificador."""

    pais: str
    perfil: str
    subtareas: List[str]
    raw: str = ""  # JSON crudo devuelto por el LLM


class PlannerAgent:
    """Agente que descompone la pregunta en subtareas y detecta jurisdicción."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()

    def plan(self, question: str) -> Plan:
        """Genera el plan de ejecución para una pregunta dada."""
        logger.info(f"[Planificador] Pregunta: {question[:80]}...")
        raw = self.llm.generate(prompt=question, system=SYSTEM_PROMPT_PLANIFICADOR)
        plan = self._parse(raw, fallback_question=question)
        logger.info(
            f"[Planificador] País={plan.pais}, perfil='{plan.perfil}', "
            f"#subtareas={len(plan.subtareas)}"
        )
        return plan

    @staticmethod
    def _parse(raw: str, fallback_question: str) -> Plan:
        """Parsea la salida JSON del LLM, con tolerancia a errores."""
        # Intentar extraer el primer bloque JSON válido
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return Plan(
                    pais=data.get("pais", "ES").upper(),
                    perfil=data.get("perfil", "contribuyente general"),
                    subtareas=data.get("subtareas", []) or [fallback_question],
                    raw=raw,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"[Planificador] JSON inválido: {e}. Usando heurística.")

        # Fallback heurístico basado en palabras clave
        pais = "DO" if re.search(
            r"\b(rep[uú]blica\s+dominicana|dgii|isr\s+dominicano|rst)\b",
            fallback_question, re.IGNORECASE
        ) else "ES"
        return Plan(
            pais=pais,
            perfil="contribuyente general",
            subtareas=[
                "Identificar el perfil fiscal del contribuyente",
                "Determinar el régimen fiscal aplicable",
                "Analizar gastos deducibles relevantes",
                "Proponer recomendaciones de cumplimiento",
            ],
            raw=raw,
        )
