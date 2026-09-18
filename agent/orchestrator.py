"""
Orquestador: decide si un mensaje es para el agente operacional, para el
de analítica, o para ninguno de los dos — gastando la menor cantidad de
tokens posible.

1. Reglas por palabras clave (gratis, no llama a ningún LLM).
2. Si es ambiguo (coinciden ambas listas o ninguna), y hay una cuenta de
   Groq dedicada al orquestador (TOKEN_GROQ2), se hace UNA llamada corta
   al LLM para desempatar (max_tokens=5).
3. Sin esa cuenta configurada, se asume "operacional" por ser el caso más
   común, salvo que no haya ninguna coincidencia de palabra clave, en
   cuyo caso se rechaza directamente.

Rechazar antes de este punto significa que ni el agente operacional ni
el de analítica gastan tokens de su propia cuenta en preguntas fuera de
tema.
"""

import re

from agent.llm_client import KEY_ORQUESTADOR, chat as llm_chat
from agent.throttle import Throttle

RECHAZO = (
    "Solo puedo ayudarte con equipos, asistencias, proveedores, ingresos, "
    "mora o afluencia del gimnasio."
)

_KEYWORDS_OPERACIONAL = [
    "equipo", "equipos", "maquina", "máquina", "maquinas", "máquinas",
    "asistencia", "asistencias", "proveedor", "proveedores", "falla",
    "fallas", "mantenimiento", "sede", "estado",
]

_KEYWORDS_ANALITICA = [
    "ingreso", "ingresos", "ganancia", "ganancias", "venta", "ventas",
    "pago", "pagos", "mora", "moroso", "morosos", "morosidad",
    "afluencia", "membresia", "membresía", "membresias", "membresías",
    "analisis", "análisis", "reporte", "reportes", "mensual", "mensuales",
    "diario", "diarios", "recaudacion", "recaudación",
]

_throttle = Throttle(min_interval=2.0)


def _match_count(text: str, keywords: list) -> int:
    return sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}", text))


def _classify_with_llm(text: str) -> str:
    if KEY_ORQUESTADOR is None:
        # Sin cuenta dedicada: si hubo AL MENOS una señal de alguna lista
        # ya se habría resuelto antes de llegar aquí, así que este caso
        # es "no matcheó nada" -> mejor rechazar que adivinar.
        return "ninguna"

    _throttle.wait()
    prompt = (
        "Eres un clasificador. Responde con UNA sola palabra, sin "
        "explicar ni puntuar: OPERACIONAL, ANALITICA o NINGUNA.\n"
        "OPERACIONAL = el mensaje pregunta por equipos/máquinas, "
        "asistencias o proveedores DE UN GIMNASIO.\n"
        "ANALITICA = el mensaje pregunta por ingresos en dinero, pagos, "
        "mora o afluencia de socios DE UN GIMNASIO.\n"
        "NINGUNA = el mensaje NO tiene relación con la operación o las "
        "finanzas de un gimnasio (geografía, clima, cultura general, "
        "otros negocios, saludos, etc).\n\n"
        f"Mensaje: {text}"
    )
    try:
        resp = llm_chat(
            KEY_ORQUESTADOR,
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        respuesta = (resp.choices[0].message.content or "").strip().upper()
    except Exception:
        return "ninguna"

    if "ANALITICA" in respuesta or "ANALÍTICA" in respuesta:
        return "analitica"
    if "OPERACIONAL" in respuesta:
        return "operacional"
    return "ninguna"


def classify(user_input: str) -> str:
    """Devuelve 'operacional', 'analitica' o 'ninguna'."""
    text = user_input.lower()
    op_score = _match_count(text, _KEYWORDS_OPERACIONAL)
    an_score = _match_count(text, _KEYWORDS_ANALITICA)

    if op_score and not an_score:
        return "operacional"
    if an_score and not op_score:
        return "analitica"
    # Ambigüedad real: ninguna coincidencia, o coincidencia en ambas listas.
    return _classify_with_llm(user_input)
