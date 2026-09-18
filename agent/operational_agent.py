"""
Agente operacional: equipos, asistencias y proveedores del gimnasio.

Usa la cuenta de Groq GROQ_API_KEY (ver llm_client.py) y el loop de
function-calling manual (ver tool_loop.py) — sin CrewAI.
"""

from agent.llm_client import KEY_OPERACIONAL
from agent.mcp_client import TOOLS_OPERACIONAL, get_tool_schemas
from agent.throttle import Throttle
from agent.tool_loop import run as run_tool_loop

_throttle = Throttle(min_interval=3.0)  # máx ~20 req/min en esta cuenta

REJECTION_MESSAGE = (
    "Solo puedo ayudarte con equipos, asistencias y proveedores del gimnasio."
)

SYSTEM_PROMPT = (
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

_TOOLS = get_tool_schemas(TOOLS_OPERACIONAL)


def process_operacional(user_input: str) -> str:
    _throttle.wait()
    return run_tool_loop(
        KEY_OPERACIONAL, SYSTEM_PROMPT, user_input, _TOOLS, REJECTION_MESSAGE
    )
