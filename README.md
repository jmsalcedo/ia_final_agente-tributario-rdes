---
title: Agente Tributario Autónomo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Asistente fiscal con RAG y bandit ε-greedy (ES + DO)
---

# 🤖 Agente Asistente Tributario Autónomo

Sistema **multiagente** basado en LLM que responde preguntas tributarias complejas para **España (IRPF / autónomos)** y **República Dominicana (ISR / RST)** usando RAG sobre normativa oficial de la AEAT y la DGII.

**Proyecto académico** — Máster en Ingeniería del Software e Inteligencia Artificial, Módulo CESA7002.

## 🏗️ Arquitectura

Sistema multiagente cooperativo con tres roles especializados:

| Agente | Función |
|--------|---------|
| 🧭 **Planificador** | Descompone la pregunta en subtareas, detecta jurisdicción ES/DO |
| 🔎 **Recuperador** | Búsqueda semántica con embeddings multilingües sobre ChromaDB |
| ✍️ **Redactor** | Sintetiza la respuesta final con razonamiento y citas |

## 🧠 Tecnologías

- **LLM:** Llama 3.1 8B Instant (vía Groq API, gratuito)
- **RAG:** ChromaDB + sentence-transformers (MiniLM multilingüe)
- **Bandit ε-greedy:** aprende qué estrategia de redacción funciona mejor
- **UI:** Streamlit
- **Despliegue:** Docker en Hugging Face Spaces

## 🔑 Configuración

Esta aplicación requiere una clave de **Groq** (gratis, sin tarjeta):

1. Crea cuenta en https://console.groq.com
2. Genera una API key (empieza por `gsk_`)
3. Añádela como **Repository secret** en este Space:
   - Settings → Variables and secrets → New secret
   - Name: `GROQ_API_KEY`
   - Value: tu clave

## 📂 Repositorio fuente

Código completo en GitHub: https://github.com/jmsalcedo/agente-tributario-autonomo

## ⚖️ Aviso legal

Este sistema es una **herramienta informativa basada en IA** y **no sustituye el asesoramiento profesional**. Consulte siempre a un asesor fiscal habilitado y a las fuentes oficiales (AEAT, DGII) antes de tomar decisiones.

## 👤 Autor

**Juan Ml. Salcedo Martínez** ([@jmsalcedo](https://github.com/jmsalcedo))