"""
Agente operacional: equipos, asistencias y proveedores del gimnasio.

Usa la cuenta de Groq GROQ_API_KEY (ver llm_config.py).
"""

import threading

from crewai import Agent

from agent.llm_config import LLM_OPERACIONAL
from agent.mcp_tools import TOOLS_OPERACIONAL, tools_for
from agent.task_runner import run_with_retries
from agent.throttle import Throttle

_agent_lock = threading.Lock()
_throttle = Throttle(min_interval=3.0)  # máx ~20 req/min en esta cuenta

EXPECTED_OUTPUT = (
    "Si es tema fuera de equipos/asistencias/proveedores, el mensaje fijo "
    "de rechazo. Si es una lista, tabla Markdown. Si es un solo dato, una "
    "frase breve en español."
)

REJECTION_MESSAGE = (
    "Solo puedo ayudarte con equipos, asistencias y proveedores del gimnasio."
)

AGENT_BACKSTORY = (
    "Asistente de gimnasio en español. SOLO respondes sobre equipos, "
    "asistencias y proveedores del gimnasio, usando las herramientas "
    "disponibles. No tienes navegador ni buscador web: no existe ninguna "
    "herramienta de búsqueda, repositorio o internet, solo las que se te "
    "dan explícitamente. No inventes ni llames herramientas que no estén "
    "en tu lista. Si la pregunta no tiene relación con esos tres temas "
    "(aunque sea sobre otro tema general), responde EXACTAMENTE esta "
    f"frase, sin agregar nada más: \"{REJECTION_MESSAGE}\" y no uses "
    "ninguna herramienta.\n\n"
    "Reglas de formato — síguelas siempre igual para que la misma consulta "
    "produzca siempre la misma forma de respuesta:\n"
    "- Si el resultado trae 2 o más elementos (lista de equipos, "
    "asistencias, proveedores, reportes, etc.), respóndelo como una tabla "
    "Markdown con solo las columnas relevantes a la pregunta.\n"
    "- Si el resultado es un solo dato, una confirmación o un conteo, "
    "respóndelo en una sola frase corta, sin tabla.\n"
    "- Nunca mezcles ambos formatos para el mismo tipo de consulta ni "
    "repitas en texto lo que ya está en la tabla.\n\n"
    "Si necesitas el ID de un equipo, primero usa `consultar_equipos`. "
    "Nunca inventes IDs ni datos: si una herramienta falla o no tienes la "
    "información, dilo brevemente. Sé conciso: prioriza claridad sobre "
    "extensión."
)


def _create_agent() -> Agent:
    return Agent(
        role="Asistente de gimnasio",
        goal=(
            "Responder de forma breve, completa y con formato consistente "
            "sobre equipos, asistencias y proveedores del gimnasio. "
            "Rechazar cualquier pregunta fuera de esos temas."
        ),
        backstory=AGENT_BACKSTORY,
        tools=tools_for(TOOLS_OPERACIONAL),
        llm=LLM_OPERACIONAL,
        verbose=True,
        max_iter=5,                # margen para: buscar ID → actuar → responder
    )


def process_operacional(user_input: str) -> str:
    _throttle.wait()
    return run_with_retries(
        _create_agent, user_input, EXPECTED_OUTPUT, _agent_lock, REJECTION_MESSAGE
    )
