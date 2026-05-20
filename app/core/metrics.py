"""Módulo de métricas y evaluación del agente.

Define la función de recompensa del bandit como combinación ponderada de:
- Relevancia: similitud coseno entre la respuesta y los chunks recuperados.
- Cobertura: fracción de subtareas mencionadas en la respuesta.
- Fundamentación: fracción de afirmaciones con cita explícita [Fuente N].

Fórmula:    R = α·Rel + β·Cob + γ·Fund,  con  α+β+γ = 1
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.agents.planner import Plan
from app.agents.writer import WriterResult
from app.core.rag import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Weights:
    """Pesos α, β, γ de la función de recompensa."""

    alpha: float = 0.45  # relevancia
    beta: float = 0.30   # cobertura
    gamma: float = 0.25  # fundamentación

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Los pesos suman {total} ≠ 1, se normalizan.")
            self.alpha /= total
            self.beta /= total
            self.gamma /= total


@dataclass
class EpisodeMetrics:
    """Métricas de un episodio completo."""

    relevance: float
    coverage: float
    grounding: float
    reward: float
    latency_s: float
    strategy: str
    pais: str
    n_chunks: int
    timestamp: str = ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno robusta."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_relevance(
    answer: str,
    chunks: List[Chunk],
    embedder,
) -> float:
    """Relevancia = similitud coseno media entre respuesta y chunks recuperados."""
    if not chunks or not answer.strip():
        return 0.0
    ans_emb = embedder.encode([answer])[0]
    chunk_embs = embedder.encode([c.text for c in chunks])
    sims = [cosine_similarity(ans_emb, ce) for ce in chunk_embs]
    return float(np.mean(sims))


def compute_coverage(answer: str, plan: Plan) -> float:
    """Cobertura = fracción de subtareas mencionadas en la respuesta.

    Heurística: contamos cuántas palabras clave de cada subtarea aparecen
    en la respuesta. Si al menos 2 sustantivos relevantes están, se considera
    cubierta. Es una proxy razonable sin requerir un evaluador LLM caro.
    """
    if not plan.subtareas:
        return 0.0
    ans_low = answer.lower()
    covered = 0
    for sub in plan.subtareas:
        keywords = [
            w for w in re.findall(r"\b[a-záéíóúñ]{5,}\b", sub.lower())
        ]
        if not keywords:
            continue
        matches = sum(1 for k in keywords if k in ans_low)
        if matches >= max(1, len(keywords) // 3):
            covered += 1
    return covered / len(plan.subtareas)


def compute_grounding(answer: str, n_chunks_available: int) -> float:
    """Fundamentación = fracción de párrafos con cita [Fuente N].

    Si la respuesta no tiene párrafos con citas, score = 0.
    Si todos los párrafos relevantes citan al menos una fuente, score = 1.
    """
    if n_chunks_available == 0:
        return 0.0
    paragraphs = [p for p in re.split(r"\n\s*\n", answer) if len(p.strip()) > 40]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if re.search(r"\[Fuente\s+\d+\]", p))
    return cited / len(paragraphs)


def reward(
    metrics: dict,
    weights: Optional[Weights] = None,
) -> float:
    """Recompensa escalar a partir de las métricas componentes."""
    w = weights or Weights()
    return (
        w.alpha * metrics.get("relevance", 0.0)
        + w.beta * metrics.get("coverage", 0.0)
        + w.gamma * metrics.get("grounding", 0.0)
    )


def log_episode_to_csv(
    metrics: EpisodeMetrics,
    csv_path: str = "logs/metrics_log.csv",
) -> None:
    """Persiste las métricas de un episodio en CSV (modo append)."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp", "strategy", "pais", "n_chunks",
                "relevance", "coverage", "grounding", "reward", "latency_s",
            ])
        writer.writerow([
            metrics.timestamp, metrics.strategy, metrics.pais, metrics.n_chunks,
            f"{metrics.relevance:.4f}",
            f"{metrics.coverage:.4f}",
            f"{metrics.grounding:.4f}",
            f"{metrics.reward:.4f}",
            f"{metrics.latency_s:.2f}",
        ])
