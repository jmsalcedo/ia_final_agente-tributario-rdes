"""Agente Asistente Tributario Autónomo — Interfaz Streamlit."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.core.orchestrator import Orchestrator
from app.core.rag import RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ===================================================================
# Configuración de la página
# ===================================================================
st.set_page_config(
    page_title="Agente Tributario Autónomo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================================
# Inicialización (cacheada)
# ===================================================================
@st.cache_resource(show_spinner="Inicializando el sistema multiagente...")
def init_orchestrator() -> Orchestrator:
    rag = RAGPipeline()
    if rag.count() == 0:
        logger.info("ChromaDB vacía. Indexando muestras automáticamente...")
        from app.core.ingest import ingest_directory
        ingest_directory(Path("data/sample"), rag)
        ingest_directory(Path("data/raw"), rag)
    return Orchestrator(rag=rag, use_bandit=True)


def detect_provider() -> tuple[str, str]:
    """Detecta el proveedor LLM activo basándose en las variables de entorno."""
    groq_token = os.getenv("GROQ_API_KEY", "")
    hf_token = os.getenv("HF_API_TOKEN", "")
    if groq_token.startswith("gsk_"):
        return "🟢 Groq (Producción)", f"Modelo: `{os.getenv('GROQ_MODEL_ID', 'llama-3.1-8b-instant')}`"
    if hf_token.startswith("hf_") and "tu_token" not in hf_token:
        return "🟢 HuggingFace (Producción)", f"Modelo: `{os.getenv('HF_MODEL_ID', 'Mistral-7B')}`"
    return "🟡 Demo", "Sin token configurado. Configure GROQ_API_KEY en .env"


# ===================================================================
# Sidebar
# ===================================================================
with st.sidebar:
    st.title("🤖 Agente Tributario")
    st.markdown(
        "**Sistema multiagente** con LLM + RAG + Bandit ε-greedy para asesoría "
        "fiscal informativa en **España** y **República Dominicana**."
    )

    st.divider()

    mode_label, mode_detail = detect_provider()
    st.markdown(f"**Modo actual:** {mode_label}")
    st.caption(mode_detail)

    if "Demo" in mode_label:
        st.warning(
            "Sin token LLM configurado. Cree cuenta gratuita en "
            "[console.groq.com](https://console.groq.com) y añada `GROQ_API_KEY` en `.env`."
        )

    st.divider()
    st.subheader("⚙️ Configuración")
    use_bandit = st.toggle("Bandit ε-greedy activo", value=True,
                           help="Si está activo, el sistema aprende qué estrategia funciona mejor.")
    forced_strategy = st.selectbox(
        "Estrategia (si bandit desactivado)",
        ["detallado", "conciso", "paso_a_paso"],
        index=0,
    )

    st.divider()
    with st.expander("ℹ️ Información del corpus"):
        try:
            rag_info = RAGPipeline()
            st.metric("Fragmentos indexados", rag_info.count())
        except Exception:
            st.metric("Fragmentos indexados", "—")

    st.divider()
    st.markdown(
        '<p style="font-size: 0.8em; color: gray;">'
        "📦 <a href='https://github.com/jmsalcedo/ia_final_agente-tributario-rdes'>"
        "GitHub</a> · Master IA · CESA7002"
        "</p>",
        unsafe_allow_html=True,
    )


# ===================================================================
# Contenido principal
# ===================================================================
st.title("🤖 Agente Asistente Tributario Autónomo")
st.markdown(
    "Asesoramiento fiscal informativo basado en normativa oficial de **AEAT** "
    "(España) y **DGII** (República Dominicana)."
)

st.warning(
    "⚖️ **Aviso legal:** Este sistema es una herramienta informativa basada en IA "
    "y **no sustituye el asesoramiento profesional**. Consulte siempre a un asesor "
    "fiscal habilitado y a las fuentes oficiales antes de tomar decisiones."
)

st.markdown("**Ejemplos de preguntas que puede formular:**")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        "- 🇪🇸 *Analizar la situación fiscal de un ingeniero de software autónomo "
        "en España e identificar gastos deducibles y estrategias de optimización.*"
    )
with col2:
    st.markdown(
        "- 🇩🇴 *Comparar el régimen ordinario con el RST de Ingresos para un "
        "consultor TI en República Dominicana.*"
    )

default_question = (
    "Analiza la situación fiscal de un ingeniero de software autónomo en España. "
    "Compara los regímenes aplicables, identifica los gastos deducibles principales "
    "y propone estrategias lícitas de optimización fiscal, incluyendo riesgos."
)

question = st.text_area(
    "💬 Tu pregunta de investigación:",
    value=default_question,
    height=120,
    help="Escribe una pregunta fiscal compleja. El agente la descompondrá en subtareas.",
)

run_button = st.button("🚀 Ejecutar agente", type="primary", use_container_width=True)


# ===================================================================
# Ejecución del agente
# ===================================================================
if run_button and question.strip():
    orchestrator = init_orchestrator()
    orchestrator.use_bandit = use_bandit
    orchestrator.fixed_strategy = forced_strategy

    progress_placeholder = st.empty()

    with progress_placeholder.container():
        with st.spinner("Procesando con el sistema multiagente..."):
            result = orchestrator.run(question)

    progress_placeholder.empty()

    with st.expander("🔍 Trazabilidad del flujo multiagente", expanded=False):
        for step in result.trace:
            st.text(step)

    st.subheader("📊 Métricas del episodio")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🎯 Relevancia", f"{result.metrics.relevance:.2f}")
    m2.metric("✅ Cobertura", f"{result.metrics.coverage:.2f}")
    m3.metric("📎 Fundamentación", f"{result.metrics.grounding:.2f}")
    m4.metric("🏆 Recompensa R", f"{result.metrics.reward:.3f}")
    m5.metric("⏱️ Latencia", f"{result.metrics.latency_s:.1f}s")

    st.caption(
        f"País: **{result.plan.pais}** · Perfil: *{result.plan.perfil}* · "
        f"Estrategia: `{result.metrics.strategy}` · "
        f"{result.metrics.n_chunks} fragmentos consultados"
    )

    with st.expander("🧭 Plan generado por el Planificador"):
        st.markdown(f"**País detectado:** {result.plan.pais}")
        st.markdown(f"**Perfil:** {result.plan.perfil}")
        st.markdown("**Subtareas:**")
        for i, st_ in enumerate(result.plan.subtareas, 1):
            st.markdown(f"{i}. {st_}")

    st.subheader("💡 Respuesta del agente")
    st.markdown(result.writer_result.answer)

    with st.expander(f"📚 Fragmentos consultados ({len(result.chunks)})"):
        for i, c in enumerate(result.chunks, 1):
            st.markdown(
                f"**[Fuente {i}]** `{c.source}` · p.{c.page} · país={c.pais}"
            )
            st.text(c.text[:500] + ("..." if len(c.text) > 500 else ""))
            st.divider()

elif run_button:
    st.error("Por favor, escribe una pregunta antes de ejecutar el agente.")


# ===================================================================
# Pie
# ===================================================================
st.markdown("---")
provider_name = "Groq · Llama-3.1-8B" if os.getenv("GROQ_API_KEY", "").startswith("gsk_") else "Mistral-7B (HF)"
st.markdown(
    f'<p style="text-align: center; color: gray; font-size: 0.85em;">'
    f"Agente Asistente Tributario Autónomo · Proyecto académico · "
    f"Modelo: <code>{provider_name}</code> · "
    f"RAG: <code>ChromaDB + MiniLM multilingüe</code>"
    f"</p>",
    unsafe_allow_html=True,
)
