"""
Config y llamadas al LLM vía litellm directo — sin el wrapper de CrewAI.

Cada agente usa una cuenta de Groq distinta para no compartir el límite
de tokens/peticiones por minuto de una sola cuenta gratuita:
- GROQ_API_KEY -> agente operacional (equipos/asistencias/proveedores).
- TOKEN_GROQ2  -> orquestador (clasifica el mensaje, solo si hay ambigüedad).
- TOKEN_GROQ3  -> agente de analítica (ingresos/mora/afluencia).

Si falta TOKEN_GROQ2 o TOKEN_GROQ3, la llave correspondiente queda en
None y quien la use debe degradar con gracia (ver orchestrator.py y
analytics_agent.py) en vez de fallar.
"""

import os

import litellm

LLM_MODEL = os.getenv("LLM_MODEL", "groq/qwen/qwen3.8-27b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

KEY_OPERACIONAL = os.getenv("GROQ_API_KEY", "").strip()
KEY_ORQUESTADOR = os.getenv("TOKEN_GROQ2", "").strip() or None
KEY_ANALITICA = os.getenv("TOKEN_GROQ3", "").strip() or None

if not KEY_OPERACIONAL:
    raise RuntimeError("GROQ_API_KEY está vacía. Revisa tu .env.")

print("─" * 60)
print("DEBUG CONFIG LLM (sin CrewAI)")
print(f"  LLM_MODEL       = {LLM_MODEL!r}")
print(f"  GROQ_API_KEY set = True  (operacional)")
print(f"  TOKEN_GROQ2 set  = {KEY_ORQUESTADOR is not None}  (orquestador)")
print(f"  TOKEN_GROQ3 set  = {KEY_ANALITICA is not None}  (analítica)")
if KEY_ORQUESTADOR is None:
    print("  ⚠️  Sin TOKEN_GROQ2: el orquestador usará solo reglas por palabra clave.")
if KEY_ANALITICA is None:
    print("  ⚠️  Sin TOKEN_GROQ3: el agente de analítica quedará deshabilitado.")
print("─" * 60)


def chat(
    api_key: str,
    messages: list,
    tools: list | None = None,
    max_tokens: int = 450,
    temperature: float = 0.2,
):
    """Llamada de chat/completion directa a Groq vía litellm."""
    return litellm.completion(
        model=LLM_MODEL,
        api_key=api_key,
        api_base=GROQ_BASE_URL,
        messages=messages,
        tools=tools or None,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=60,
        num_retries=3,
    )
