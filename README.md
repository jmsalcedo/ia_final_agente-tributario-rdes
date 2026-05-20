# 🤖 Agente Asistente Tributario Español y Dominicano

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.36-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-0.2-green.svg)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange.svg)](https://groq.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)

> Sistema **multiagente** basado en LLM que responde preguntas tributarias complejas para **España (IRPF / autónomos)** y **República Dominicana (ISR / RST)** usando RAG sobre normativa oficial de la AEAT y la DGII.

**Proyecto académico** — Máster en Ingeniería del Software e Inteligencia Artificial, Módulo CESA7002.

---

## 🌐 Demo en vivo

🔗 **Aplicación desplegada:** `https://ia-final-agente-tributario-rdes.onrender.com`

📂 **Repositorio:** https://github.com/jmsalcedo/ia_final_agente-tributario-rdes

---

## 🏗️ Arquitectura

Sistema **multiagente cooperativo** con tres roles especializados:

```
   Usuario
      │
      ▼
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│             Orquestador (LangGraph)                │
│                                                    │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐   │
│  │Planifica-│──▶│Recuperador │──▶│  Redactor  │   │
│  │   dor    │   │   (RAG)    │   │   (LLM)    │   │
│  └──────────┘   └─────┬──────┘   └─────┬──────┘   │
│                       │                 │          │
│                       ▼                 ▼          │
│                  ┌────────┐       ┌──────────┐    │
│                  │Chroma  │       │ Groq API │    │
│                  │ DB     │       │ Llama3.1 │    │
│                  └────────┘       └──────────┘    │
└────────────────────────────────────────────────────┘
         │
         ▼
   Evaluador + Bandit ε-greedy (mejora iterativa)
```

| Agente | Función |
|--------|---------|
| 🧭 **Planificador** | Descompone la pregunta en subtareas, detecta jurisdicción ES/DO |
| 🔎 **Recuperador** | Búsqueda semántica con embeddings multilingües sobre ChromaDB |
| ✍️ **Redactor** | Sintetiza la respuesta final con razonamiento y citas |

---

## 🚀 Inicio rápido

### Opción A — Con Docker (recomendada)

```bash
git clone https://github.com/jmsalcedo/ia_final_agente-tributario-rdes.git
cd ia_final_agente-tributario-rdes
cp .env.example .env
# Edita .env y añade tu GROQ_API_KEY (ver más abajo cómo obtenerla)
docker-compose up --build
```

Abre tu navegador en **http://localhost:8501**

### Opción B — Sin Docker (entorno local)

```bash
git clone https://github.com/jmsalcedo/ia_final_agente-tributario-rdes.git
cd ia_final_agente-tributario-rdes
python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Edita el token
python -m app.core.ingest          # Indexar corpus (primera vez)
streamlit run app/streamlit_app.py
```

---

## 🔑 Obtener clave de Groq (gratis)

**Groq** es el proveedor LLM principal del proyecto. Ofrece **Llama 3.1 8B Instant** gratis:
- ⚡ ~300 tokens/segundo (3-10× más rápido que otros)
- 🆓 14.400 requests/día sin tarjeta de crédito
- 🚀 Compatible OpenAI

**Pasos para obtenerla:**
1. Crea cuenta en https://console.groq.com (con Google o GitHub)
2. Ve a **API Keys** → **Create API Key**
3. Copia la clave (empieza por `gsk_...`)
4. Pégala en tu archivo `.env`:
   ```
   GROQ_API_KEY=gsk_tu_clave_aqui
   ```

### Alternativa: Hugging Face (fallback opcional)

Si prefieres HuggingFace, el sistema lo detecta automáticamente si configuras `HF_API_TOKEN` en `.env`. Sin embargo, **desde 2025 el tier gratuito de HF está limitado** a modelos pequeños y embeddings; Groq es la opción recomendada.

---

## 📚 Corpus documental

El sistema indexa normativa fiscal oficial:

**🇪🇸 España (AEAT):**
- Manual Práctico Renta 2024 (Tomos 1 y 2)

**🇩🇴 República Dominicana (DGII):**
- Código Tributario (Ley 11-92)
- Norma General Régimen Simplificado de Tributación (RST)

Para cargar el corpus completo:

```bash
python scripts/download_corpus.py
python -m app.core.ingest
```

> ⚡ El proyecto incluye **muestras pre-procesadas** en `data/sample/` para que puedas probar sin descargar nada.

---

## 🧪 Ejecutar experimentos

```bash
jupyter notebook notebook/agente_tributario.ipynb
```

El notebook reproduce todos los experimentos: comparativa baseline vs política aprendida, métricas, gráficos.

---

## 🐳 Despliegue en Render

1. Haz fork de este repo a tu cuenta GitHub.
2. Entra a https://dashboard.render.com → **New** → **Web Service**.
3. Conecta tu repositorio.
4. Render detectará automáticamente `render.yaml`.
5. Añade la variable de entorno `GROQ_API_KEY` en el panel de Render.
6. Pulsa **Create Web Service**.

⚠️ El plan gratuito de Render tiene "cold start" (~30s la primera carga).

---

## 📂 Estructura del proyecto

```
ia_final_agente-tributario-rdes/
├── app/
│   ├── streamlit_app.py           # Interfaz Streamlit
│   ├── agents/                    # 3 agentes especializados
│   ├── core/                      # Orquestador, RAG, LLM, bandit, métricas
│   └── utils/                     # Logger
├── data/
│   ├── raw/                       # PDFs oficiales descargados
│   ├── processed/                 # Texto extraído + chunks
│   └── sample/                    # Muestras pre-procesadas
├── notebook/
│   └── agente_tributario.ipynb    # Notebook con experimentos
├── informe/
│   ├── informe_tecnico.tex        # Informe LaTeX (Overleaf)
│   ├── referencias.bib            # Bibliografía APA
│   └── figs/                      # Gráficos generados
├── tests/                         # Tests unitarios
├── scripts/
│   └── download_corpus.py         # Descarga de PDFs oficiales
├── Dockerfile
├── docker-compose.yml
├── render.yaml
└── requirements.txt
```

---

## ⚖️ Aviso legal

> Este sistema es una **herramienta informativa basada en IA** y **no sustituye el asesoramiento profesional** de un asesor fiscal habilitado. Los autores no se hacen responsables de las decisiones tomadas a partir de sus respuestas. Consulte siempre a un profesional colegiado y a las fuentes oficiales (AEAT, DGII).

---

## 📜 Licencia

CESTE © 2026 — Proyecto académico para el módulo CESA7002.

## 👤 Autor

**Juan Ml. Salcedo Martínez** ([@jmsalcedo](https://github.com/jmsalcedo))
