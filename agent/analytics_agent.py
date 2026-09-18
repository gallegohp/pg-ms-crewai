"""
Agente de analítica: ingresos (dinero), mora y afluencia (entradas
físicas) del gimnasio.

Usa la cuenta de Groq TOKEN_GROQ3 (ver llm_config.py). Si esa cuenta no
está configurada, el agente queda deshabilitado y lo dice explícitamente
en vez de fallar.
"""

import threading

from crewai import Agent

from agent.llm_config import LLM_ANALITICA
from agent.mcp_tools import TOOLS_ANALITICA, tools_for
from agent.task_runner import run_with_retries
from agent.throttle import Throttle

_agent_lock = threading.Lock()
_throttle = Throttle(min_interval=3.0)  # máx ~20 req/min en esta cuenta

EXPECTED_OUTPUT = (
    "Si es tema fuera de ingresos/mora/afluencia, el mensaje fijo de "
    "rechazo. Si es una lista, tabla Markdown. Si es un solo dato, una "
    "frase breve en español."
)

REJECTION_MESSAGE = "Solo puedo ayudarte con ingresos, mora y afluencia del gimnasio."

AGENT_BACKSTORY = (
    "Analista de datos de un gimnasio, en español. SOLO respondes sobre "
    "ingresos (dinero de pagos/membresías), mora y afluencia (entradas "
    "físicas de socios) del gimnasio, usando las herramientas disponibles. "
    "No tienes navegador ni buscador web: no existe ninguna herramienta de "
    "búsqueda, repositorio o internet, solo las que se te dan "
    "explícitamente. No inventes ni llames herramientas que no estén en tu "
    "lista. Si la pregunta no tiene relación con esos temas, responde "
    f"EXACTAMENTE esta frase, sin agregar nada más: \"{REJECTION_MESSAGE}\" "
    "y no uses ninguna herramienta.\n\n"
    "Distingue bien los dos sentidos de 'ingreso': dinero recibido (tools "
    "de reporte de ingresos) contra entradas físicas al gimnasio "
    "(afluencia/check-ins). No los confundas ni mezcles sus datos.\n\n"
    "Si no te dan una fecha, asume la de hoy. Si una herramienta falla por "
    "permisos (error 403 o similar), dilo tal cual en vez de inventar "
    "cifras.\n\n"
    "Reglas de formato — síguelas siempre igual para que la misma consulta "
    "produzca siempre la misma forma de respuesta:\n"
    "- Si el resultado trae 2 o más elementos, respóndelo como una tabla "
    "Markdown con solo las columnas relevantes a la pregunta.\n"
    "- Si el resultado es un solo dato, un total o una confirmación, "
    "respóndelo en una sola frase corta, sin tabla.\n"
    "- Nunca mezcles ambos formatos para el mismo tipo de consulta ni "
    "repitas en texto lo que ya está en la tabla.\n\n"
    "Nunca inventes cifras. Sé conciso: prioriza claridad sobre extensión."
)


def _create_agent() -> Agent:
    return Agent(
        role="Analista de datos del gimnasio",
        goal=(
            "Responder de forma breve, completa y con formato consistente "
            "sobre ingresos, mora y afluencia del gimnasio. Rechazar "
            "cualquier pregunta fuera de esos temas."
        ),
        backstory=AGENT_BACKSTORY,
        tools=tools_for(TOOLS_ANALITICA),
        llm=LLM_ANALITICA,
        verbose=True,
        max_iter=5,
    )


def process_analitica(user_input: str) -> str:
    if LLM_ANALITICA is None:
        return (
            "El agente de analítica no está disponible todavía: falta "
            "configurar TOKEN_GROQ3 en el .env."
        )

    _throttle.wait()
    return run_with_retries(
        _create_agent, user_input, EXPECTED_OUTPUT, _agent_lock, REJECTION_MESSAGE
    )
