"""Tests unitarios del Agente Planificador."""
from __future__ import annotations

import os

import pytest

# Forzar modo demo para tests (no necesitan token)
os.environ["APP_MODE"] = "demo"

from app.agents.planner import PlannerAgent, Plan
from app.core.llm_client import LLMClient


def test_plan_devuelve_objeto_plan():
    """El planner debe devolver un Plan con país, perfil y subtareas."""
    agent = PlannerAgent(llm=LLMClient())
    plan = agent.plan("¿Cómo tributa un autónomo en España?")
    assert isinstance(plan, Plan)
    assert plan.pais in {"ES", "DO"}
    assert plan.perfil
    assert len(plan.subtareas) >= 1


def test_detecta_pais_dominicana_heuristica():
    """Si el LLM falla, la heurística debe detectar DO por palabras clave."""
    raw = "esto no es JSON válido"
    plan = PlannerAgent._parse(raw, "Soy un consultor en República Dominicana, ¿qué impuestos pago?")
    assert plan.pais == "DO"


def test_detecta_pais_españa_por_defecto():
    """Sin pistas, el país por defecto es ES."""
    raw = "esto no es JSON válido"
    plan = PlannerAgent._parse(raw, "¿Qué gastos puedo deducir?")
    assert plan.pais == "ES"


def test_parsea_json_correcto():
    """JSON bien formado debe ser parseado."""
    raw = '{"pais": "DO", "perfil": "freelance TI", "subtareas": ["a", "b"]}'
    plan = PlannerAgent._parse(raw, "pregunta")
    assert plan.pais == "DO"
    assert plan.perfil == "freelance TI"
    assert plan.subtareas == ["a", "b"]
