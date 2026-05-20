"""Cliente LLM unificado.

Soporta tres modos con selección automática:
- `groq`: Llama 3.1 8B Instant a 800 tok/s.
- `huggingface`: HuggingFace Inference API (limitado en gratuito desde 2025).
- `demo`: respuestas mock si no hay token configurado.

Orden de preferencia automática: Groq > HuggingFace > demo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuración del cliente LLM."""

    provider: str = "auto"   # "auto" | "groq" | "huggingface" | "demo"
    model_id: str = "llama-3.1-8b-instant"
    temperature: float = 0.3
    max_tokens: int = 800
    groq_api_key: Optional[str] = None
    hf_api_token: Optional[str] = None
    hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"


class LLMClient:
    """Cliente LLM con selección automática de proveedor."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "auto"),
            model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            hf_api_token=os.getenv("HF_API_TOKEN"),
            hf_model_id=os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
        )
        self.mode = self._determine_mode()
        self.client = None
        self._initialize_client()

    def _determine_mode(self) -> str:
        """Decide el proveedor a usar."""
        if os.getenv("APP_MODE", "production").lower() == "demo":
            return "demo"

        if self.config.provider in ("groq", "huggingface", "demo"):
            if self.config.provider == "groq" and not self._has_groq_token():
                logger.warning("Forzado a 'groq' pero falta GROQ_API_KEY. Uso demo.")
                return "demo"
            if self.config.provider == "huggingface" and not self._has_hf_token():
                logger.warning("Forzado a 'huggingface' pero falta HF_API_TOKEN. Uso demo.")
                return "demo"
            return self.config.provider

        # Selección automática
        if self._has_groq_token():
            return "groq"
        if self._has_hf_token():
            return "huggingface"
        return "demo"

    def _has_groq_token(self) -> bool:
        return bool(self.config.groq_api_key) and self.config.groq_api_key.startswith("gsk_")

    def _has_hf_token(self) -> bool:
        return (
            bool(self.config.hf_api_token)
            and self.config.hf_api_token.startswith("hf_")
            and "tu_token" not in self.config.hf_api_token
        )

    def _initialize_client(self) -> None:
        """Inicializa el cliente del proveedor seleccionado."""
        if self.mode == "groq":
            try:
                from groq import Groq
                self.client = Groq(api_key=self.config.groq_api_key)
                logger.info(
                    f"LLMClient inicializado en modo GROQ ({self.config.model_id})"
                )
            except ImportError:
                logger.error(
                    "Paquete 'groq' no instalado. Ejecuta: pip install groq"
                )
                self.mode = "demo"
            except Exception as e:
                logger.error(f"Error al inicializar Groq: {e}. Cayendo a demo.")
                self.mode = "demo"

        elif self.mode == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(
                    model=self.config.hf_model_id,
                    token=self.config.hf_api_token,
                )
                logger.info(
                    f"LLMClient inicializado en modo HUGGINGFACE ({self.config.hf_model_id})"
                )
            except Exception as e:
                logger.error(f"Error al inicializar HF: {e}. Cayendo a demo.")
                self.mode = "demo"

        if self.mode == "demo":
            logger.warning(
                "LLMClient en modo DEMO. Configure GROQ_API_KEY en .env "
                "(gratis en https://console.groq.com)"
            )

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Genera texto a partir de un prompt.

        Args:
            prompt: Pregunta o instrucción del usuario.
            system: Instrucción de sistema opcional.

        Returns:
            Texto generado por el modelo.
        """
        if self.mode == "demo":
            return self._mock_response(prompt, system)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            if self.mode == "groq":
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.config.model_id,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content.strip()

            elif self.mode == "huggingface":
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error en llamada al LLM ({self.mode}): {e}. Cayendo a demo.")
            return self._mock_response(prompt, system)

        return self._mock_response(prompt, system)

    @staticmethod
    def _mock_response(prompt: str, system: Optional[str]) -> str:
        """Respuesta plantilla en modo demo."""
        if system and "planificador" in system.lower():
            return (
                '{"pais": "ES", "perfil": "contribuyente general", "subtareas": ['
                '"Identificar el perfil del contribuyente",'
                '"Determinar el régimen fiscal aplicable",'
                '"Analizar gastos deducibles",'
                '"Proponer estrategias de optimización fiscal"]}'
            )
        if system and "redactor" in system.lower():
            return (
                "[MODO DEMO — respuesta plantilla]\n\n"
                "Esta es una respuesta generada en modo demostración porque no se ha "
                "configurado un token LLM válido. La estructura del agente sí está "
                "ejecutándose correctamente: planificación, recuperación y agregación "
                "funcionan. Para obtener respuestas reales del LLM, configure "
                "GROQ_API_KEY en su archivo .env (gratis en https://console.groq.com).\n\n"
                "**Aviso legal:** Esta herramienta es informativa y no sustituye al asesor fiscal."
            )
        return "[MODO DEMO] Configure GROQ_API_KEY en .env para activar el LLM real."