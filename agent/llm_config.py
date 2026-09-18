"""
Construcción de los LLM usados por los distintos agentes.

Cada agente usa una cuenta de Groq distinta para no compartir el límite
de tokens/peticiones por minuto de una sola cuenta gratuita:
- GROQ_API_KEY -> agente operacional (equipos/asistencias/proveedores).
- TOKEN_GROQ2  -> orquestador (clasifica el mensaje, solo si hay ambigüedad).
- TOKEN_GROQ3  -> agente de analítica (ingresos/mora/afluencia).

Si falta TOKEN_GROQ2 o TOKEN_GROQ3, el LLM correspondiente queda en None
y quien lo use debe degradar con gracia (ver orchestrator.py y
analytics_agent.py) en vez de fallar.

IMPORTANTE: groq/compound y groq/compound-mini NO soportan tools
personalizadas según la documentación oficial de Groq. Por eso se usa
un modelo estándar con tool-calling real: groq/openai/gpt-oss-120b.
"""

import os

from crewai import LLM

LLM_MODEL = os.getenv("LLM_MODEL", "groq/openai/gpt-oss-120b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()


def _build_llm(api_key: str, *, max_tokens: int = 450, temperature: float = 0.2) -> LLM:
    if LLM_MODEL.startswith("groq/"):
        return LLM(
            model=LLM_MODEL,                  # ej: "groq/openai/gpt-oss-120b"
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=10,
            timeout=60,
        )
    if LLM_MODEL.startswith("gemini/"):
        return LLM(model=LLM_MODEL, api_key=api_key, max_tokens=max_tokens, temperature=temperature)
    return LLM(model=LLM_MODEL, api_key=api_key, max_tokens=max_tokens, temperature=temperature)


_groq_operacional = os.getenv("GROQ_API_KEY", "").strip()
if not _groq_operacional:
    raise RuntimeError("GROQ_API_KEY está vacía. Revisa tu .env.")

# ── Cuenta 1: agente operacional ──
LLM_OPERACIONAL = _build_llm(_groq_operacional)

# ── Cuenta 2: orquestador ──
# OJO: gpt-oss-120b es un modelo "razonador": consume tokens pensando
# ANTES de escribir la respuesta final, y esos tokens de razonamiento
# cuentan contra max_tokens. Con un valor chico (ej. 5-40) el modelo se
# queda a mitad de razonar y devuelve contenido vacío. 200 le da margen
# de sobra para razonar y aun así devolver la palabra de clasificación;
# sigue siendo mucho más barato que una respuesta completa (450).
_groq_orquestador = os.getenv("TOKEN_GROQ2", "").strip()
LLM_ORQUESTADOR = _build_llm(_groq_orquestador, max_tokens=200, temperature=0.0) if _groq_orquestador else None

# ── Cuenta 3: agente de analítica ──
_groq_analitica = os.getenv("TOKEN_GROQ3", "").strip()
LLM_ANALITICA = _build_llm(_groq_analitica) if _groq_analitica else None

print("─" * 60)
print("DEBUG CONFIG LLM (multi-agente)")
print(f"  LLM_PROVIDER            = {LLM_PROVIDER!r}")
print(f"  LLM_MODEL               = {LLM_MODEL!r}")
print(f"  GROQ_API_KEY set        = {bool(_groq_operacional)}  (operacional)")
print(f"  TOKEN_GROQ2 set         = {bool(_groq_orquestador)}  (orquestador)")
print(f"  TOKEN_GROQ3 set         = {bool(_groq_analitica)}  (analítica)")
if not _groq_orquestador:
    print("  ⚠️  Sin TOKEN_GROQ2: el orquestador usará solo reglas por palabra clave.")
if not _groq_analitica:
    print("  ⚠️  Sin TOKEN_GROQ3: el agente de analítica quedará deshabilitado.")
print("─" * 60)
