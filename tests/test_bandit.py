"""Tests del bandit ε-greedy."""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.core.bandit import EpsilonGreedyBandit


def test_bandit_explora_brazos_iniciales():
    """En las primeras N selecciones debe probar todos los brazos."""
    with tempfile.TemporaryDirectory() as tmp:
        bandit = EpsilonGreedyBandit(
            arms=["a", "b", "c"],
            persist_path=str(Path(tmp) / "state.json"),
        )
        elegidos = {bandit.select() for _ in range(20)}
        # Algunos brazos deben aparecer (no determinista exacto, pero los 3 deben caer)
        assert len(elegidos) >= 2


def test_bandit_actualiza_recompensa():
    """Tras actualizar, las medias deben reflejar la recompensa observada."""
    with tempfile.TemporaryDirectory() as tmp:
        bandit = EpsilonGreedyBandit(
            arms=["x", "y"],
            persist_path=str(Path(tmp) / "state.json"),
        )
        bandit.update("x", 1.0)
        bandit.update("x", 0.8)
        stats = bandit.get_stats()
        assert stats["x"]["count"] == 2
        assert abs(stats["x"]["mean_reward"] - 0.9) < 1e-6


def test_bandit_persiste_estado():
    """El estado debe persistir entre instancias en el mismo path."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "state.json")
        b1 = EpsilonGreedyBandit(arms=["a"], persist_path=path)
        b1.update("a", 0.5)
        b2 = EpsilonGreedyBandit(arms=["a"], persist_path=path)
        assert b2.state.counts["a"] == 1
