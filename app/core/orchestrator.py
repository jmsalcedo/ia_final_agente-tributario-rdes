"""Orquestador multiagente.

Coordina el flujo: Planificador → Recuperador → Redactor → Evaluador → Bandit.

Devuelve un objeto EpisodeResult con toda la trazabilidad para la UI
y los experimentos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from app.agents.planner import Plan, PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.writer import STRATEGIES, WriterAgent, WriterResult
from app.core.bandit import EpsilonGreedyBandit
from app.core.llm_client import LLMClient
from app.core.metrics import (
    EpisodeMetrics,
    Weights,
    compute_coverage,
    compute_grounding,
    compute_relevance,
    log_episode_to_csv,
    reward,
)
from app.core.rag import Chunk, RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeResult:
    """Resultado completo de un episodio del agente."""

    question: str
    plan: Plan
    chunks: List[Chunk]
    writer_result: WriterResult
    metrics: EpisodeMetrics
    trace: List[str] = field(default_factory=list)  # paso a paso para la UI


class Orchestrator:
    """Orquesta los 3 agentes y la política de aprendizaje."""

    def __init__(
        self,
        rag: Optional[RAGPipeline] = None,
        llm: Optional[LLMClient] = None,
        bandit: Optional[EpsilonGreedyBandit] = None,
        use_bandit: bool = True,
        fixed_strategy: str = "detallado",
        weights: Optional[Weights] = None,
    ) -> None:
        self.rag = rag or RAGPipeline()
        self.llm = llm or LLMClient()
        self.planner = PlannerAgent(self.llm)
        self.retriever = RetrieverAgent(self.rag)
        self.writer = WriterAgent(self.llm)
        self.bandit = bandit or EpsilonGreedyBandit(arms=STRATEGIES, epsilon=0.2)
        self.use_bandit = use_bandit
        self.fixed_strategy = fixed_strategy
        self.weights = weights or Weights()

    def run(self, question: str) -> EpisodeResult:
        """Ejecuta un episodio completo y devuelve resultados + métricas."""
        trace = []
        start = time.time()

        # --- Paso 1: Planificación ---
        trace.append("🧭 Planificador: analizando la pregunta...")
        plan = self.planner.plan(question)
        trace.append(
            f"   ✓ País detectado: {plan.pais} | "
            f"Perfil: {plan.perfil} | "
            f"{len(plan.subtareas)} subtareas"
        )

        # --- Paso 2: Recuperación ---
        trace.append("🔎 Recuperador: consultando ChromaDB...")
        chunks = self.retriever.retrieve_for_plan(plan, question)
        trace.append(f"   ✓ {len(chunks)} fragmentos relevantes recuperados")

        # --- Paso 3: Selección de estrategia ---
        if self.use_bandit:
            strategy = self.bandit.select()
            trace.append(f"🎲 Bandit: estrategia seleccionada = '{strategy}'")
        else:
            strategy = self.fixed_strategy
            trace.append(f"📌 Estrategia fija: '{strategy}'")

        # --- Paso 4: Redacción ---
        trace.append("✍️ Redactor: sintetizando la respuesta final...")
        writer_result = self.writer.write(question, plan, chunks, strategy=strategy)
        trace.append(f"   ✓ Respuesta generada ({len(writer_result.answer)} caracteres)")

        # --- Paso 5: Evaluación ---
        trace.append("📊 Evaluador: calculando métricas...")
        relevance = compute_relevance(writer_result.answer, chunks, self.rag.embedder)
        coverage = compute_coverage(writer_result.answer, plan)
        grounding = compute_grounding(writer_result.answer, len(chunks))
        r = reward(
            {"relevance": relevance, "coverage": coverage, "grounding": grounding},
            self.weights,
        )
        latency = time.time() - start

        metrics = EpisodeMetrics(
            relevance=relevance,
            coverage=coverage,
            grounding=grounding,
            reward=r,
            latency_s=latency,
            strategy=strategy,
            pais=plan.pais,
            n_chunks=len(chunks),
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        trace.append(
            f"   ✓ R={r:.3f} | Rel={relevance:.2f} Cob={coverage:.2f} Fund={grounding:.2f} "
            f"| {latency:.1f}s"
        )

        # --- Paso 6: Actualizar bandit ---
        if self.use_bandit:
            self.bandit.update(
                strategy,
                r,
                metadata={
                    "pais": plan.pais,
                    "relevance": round(relevance, 3),
                    "coverage": round(coverage, 3),
                    "grounding": round(grounding, 3),
                },
            )
            trace.append("🔁 Bandit actualizado con la recompensa observada")

        # --- Paso 7: Persistir métricas ---
        log_episode_to_csv(metrics)

        return EpisodeResult(
            question=question,
            plan=plan,
            chunks=chunks,
            writer_result=writer_result,
            metrics=metrics,
            trace=trace,
        )
