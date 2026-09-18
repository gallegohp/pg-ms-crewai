"""
Módulo de Agente Conversacional con CrewAI y MCP.

Este __init__ prepara el entorno UNA sola vez (limpieza de variables de
Google, flags de CrewAI, carga del .env y parche de LiteLLM) antes de que
cualquier submódulo (llm_config, mcp_tools, operational_agent,
analytics_agent) importe `crewai`. El orden importa: CrewAI lee algunas
de estas variables de entorno al importarse.
"""

import os

# ── Limpiar variables de Google ──
for _var in (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
):
    os.environ.pop(_var, None)

# ── Forzar LiteLLM ──
os.environ["CREWAI_LLM_USE_LITELLM"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path), override=False)

# Segunda limpieza por si el .env las reintrodujo
for _var in (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
):
    os.environ.pop(_var, None)

# ── Configurar LiteLLM para reintentos agresivos ──
import litellm
litellm.num_retries = 10
litellm.request_timeout = 60

# Parche de compatibilidad con Groq
try:
    from agent.litellm_patch import apply_patch
    apply_patch()
except ImportError:
    pass
