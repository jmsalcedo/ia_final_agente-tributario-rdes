"""Tests del módulo de métricas."""
from __future__ import annotations

import numpy as np

from app.agents.planner import Plan
from app.core.metrics import (
    Weights,
    compute_coverage,
    compute_grounding,
    cosine_similarity,
    reward,
)


def test_cosine_similarity_basico():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

    c = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, c)) < 1e-6


def test_cobertura_cero_si_no_hay_subtareas():
    plan = Plan(pais="ES", perfil="x", subtareas=[])
    assert compute_coverage("texto", plan) == 0.0


def test_cobertura_detecta_subtareas_cubiertas():
    plan = Plan(
        pais="ES",
        perfil="autonomo",
        subtareas=["identificar gastos deducibles", "analizar el regimen fiscal"],
    )
    respuesta = (
        "Los gastos deducibles para un autónomo incluyen suministros. "
        "El régimen fiscal aplicable es la estimación directa."
    )
    cov = compute_coverage(respuesta, plan)
    assert cov > 0.5


def test_grounding_sin_citas_es_cero():
    respuesta = "Texto sin citas explícitas a fuentes."
    assert compute_grounding(respuesta, n_chunks_available=3) == 0.0


def test_grounding_con_citas_correctas():
    respuesta = (
        "Primera afirmación importante con respaldo [Fuente 1].\n\n"
        "Segunda afirmación que también cita [Fuente 2]."
    )
    g = compute_grounding(respuesta, n_chunks_available=3)
    assert g == 1.0


def test_reward_combina_pesos():
    w = Weights(alpha=0.5, beta=0.3, gamma=0.2)
    r = reward({"relevance": 1.0, "coverage": 1.0, "grounding": 1.0}, w)
    assert abs(r - 1.0) < 1e-6

    r2 = reward({"relevance": 0.0, "coverage": 0.0, "grounding": 0.0}, w)
    assert r2 == 0.0
