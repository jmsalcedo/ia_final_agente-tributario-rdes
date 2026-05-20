"""Política Bandit ε-greedy (mecanismo de RL ligero).

Aprende a seleccionar la mejor estrategia de prompting del Redactor
basándose en la recompensa observada en episodios previos.

Implementa el clásico problema de "multi-armed bandit":
- Brazos (acciones): las estrategias del Redactor ("conciso", "detallado", "paso_a_paso")
- Recompensa: R = α·relevancia + β·cobertura + γ·fundamentación
- Política: con probabilidad ε explora (acción aleatoria), con 1-ε explota
  (la acción de mayor recompensa media).

Referencia: Sutton & Barto (2018), Reinforcement Learning, capítulo 2.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BanditState:
    """Estado del bandit: cuenta de selecciones y recompensa media por brazo."""

    counts: Dict[str, int] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)  # acumulada
    history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "counts": self.counts,
            "rewards": self.rewards,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BanditState":
        return cls(
            counts=data.get("counts", {}),
            rewards=data.get("rewards", {}),
            history=data.get("history", []),
        )


class EpsilonGreedyBandit:
    """Política ε-greedy con persistencia en disco."""

    def __init__(
        self,
        arms: List[str],
        epsilon: float = 0.2,
        persist_path: Optional[str] = "logs/bandit_state.json",
    ) -> None:
        self.arms = arms
        self.epsilon = epsilon
        self.persist_path = Path(persist_path) if persist_path else None
        self.state = self._load_or_init()

    def _load_or_init(self) -> BanditState:
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Bandit cargado desde {self.persist_path}")
                return BanditState.from_dict(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"No se pudo cargar estado del bandit: {e}")
        state = BanditState()
        for arm in self.arms:
            state.counts[arm] = 0
            state.rewards[arm] = 0.0
        return state

    def _persist(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def select(self) -> str:
        """Selecciona un brazo según la política ε-greedy."""
        # Si algún brazo no se ha probado nunca, forzar exploración inicial
        unexplored = [a for a in self.arms if self.state.counts.get(a, 0) == 0]
        if unexplored:
            choice = random.choice(unexplored)
            logger.info(f"[Bandit] Exploración inicial → '{choice}'")
            return choice

        if random.random() < self.epsilon:
            choice = random.choice(self.arms)
            logger.info(f"[Bandit] Exploración (ε={self.epsilon}) → '{choice}'")
            return choice

        # Explotación: brazo con mayor recompensa media
        means = {a: self.state.rewards[a] / max(self.state.counts[a], 1) for a in self.arms}
        choice = max(means, key=means.get)
        logger.info(
            f"[Bandit] Explotación → '{choice}' (media={means[choice]:.3f})"
        )
        return choice

    def update(self, arm: str, reward: float, metadata: Optional[dict] = None) -> None:
        """Actualiza el estado tras observar una recompensa."""
        if arm not in self.arms:
            logger.warning(f"Brazo desconocido: {arm}")
            return
        self.state.counts[arm] = self.state.counts.get(arm, 0) + 1
        self.state.rewards[arm] = self.state.rewards.get(arm, 0.0) + reward
        self.state.history.append(
            {
                "episode": len(self.state.history) + 1,
                "arm": arm,
                "reward": round(reward, 4),
                **(metadata or {}),
            }
        )
        self._persist()
        logger.info(
            f"[Bandit] Actualizado '{arm}': nueva media = "
            f"{self.state.rewards[arm] / self.state.counts[arm]:.3f} "
            f"(n={self.state.counts[arm]})"
        )

    def get_stats(self) -> Dict[str, dict]:
        """Devuelve estadísticas por brazo."""
        return {
            a: {
                "count": self.state.counts.get(a, 0),
                "mean_reward": (
                    self.state.rewards.get(a, 0.0) / max(self.state.counts.get(a, 1), 1)
                ),
            }
            for a in self.arms
        }

    def reset(self) -> None:
        """Reinicia el estado del bandit."""
        self.state = BanditState()
        for arm in self.arms:
            self.state.counts[arm] = 0
            self.state.rewards[arm] = 0.0
        self._persist()
        logger.info("Bandit reiniciado.")
