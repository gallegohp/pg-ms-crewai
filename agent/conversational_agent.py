"""
Agente Conversacional con CrewAI + MCP.
Optimizado para reducir el consumo de tokens del LLM.

IMPORTANTE: groq/compound y groq/compound-mini NO soportan tools
personalizadas (custom function-calling) según la documentación oficial
de Groq — solo pueden usar sus propias tools internas (web search, code
execution, etc). Por eso no se usan aquí: el modelo nunca llegaba a
invocar las tools MCP y terminaba inventando respuestas. Se usa en su
lugar un modelo estándar de Groq con tool-calling real y soporte
confirmado en esta cuenta: groq/openai/gpt-oss-120b.
"""

import os
import time
import threading

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

_agent_lock = threading.Lock()

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

from crewai import Agent, Task, LLM
from crewai.mcp import MCPServerSSE
from crewai.mcp.filters import create_static_tool_filter

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

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE HERRAMIENTAS MCP (FILTRADAS)
# ─────────────────────────────────────────────────────────────
# Solo cargamos las herramientas más relevantes para reducir
# el esquema JSON que CrewAI envía al LLM en cada request.

TOOLS_RELEVANTES = [
    # Equipos
    "consultar_equipos",
    "contar_equipos_por_estado",
    "cambiar_estado_equipo",
    "reportar_falla_equipo",
    # Asistencias
    "consultar_asistencias",
    "contar_asistencias_por_fecha",
    # Proveedores
    "consultar_proveedores",
]

mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

mcp_server = MCPServerSSE(
    url=mcp_server_url,
    tool_filter=create_static_tool_filter(
        allowed_tool_names=TOOLS_RELEVANTES
    ),
    cache_tools_list=True,
)

# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────
llm_model = os.getenv("LLM_MODEL", "groq/openai/gpt-oss-120b")
llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()

print("─" * 60)
print("DEBUG CONFIG LLM")
print(f"  LLM_PROVIDER           = {llm_provider!r}")
print(f"  LLM_MODEL              = {llm_model!r}")
print(f"  GROQ_API_KEY set       = {bool(os.getenv('GROQ_API_KEY'))}")
print(f"  CREWAI_LLM_USE_LITELLM = {os.getenv('CREWAI_LLM_USE_LITELLM')}")
print(f"  TOOLS_FILTRADAS        = {len(TOOLS_RELEVANTES)}")
print("─" * 60)

if llm_model.startswith("groq/"):
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY está vacía. Revisa tu .env.")
    llm = LLM(
        model=llm_model,                      # ej: "groq/openai/gpt-oss-120b"
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        max_tokens=450,     # margen para no cortar tablas/listas a mitad de respuesta
        temperature=0.2,    # baja variabilidad: misma consulta -> mismo formato de respuesta
        max_retries=10,
        timeout=60,
    )
elif llm_model.startswith("gemini/"):
    llm = LLM(
        model=llm_model,
        api_key=os.getenv("GEMINI_API_KEY", ""),
        max_tokens=450,
        temperature=0.2,
    )
else:
    llm = LLM(
        model=llm_model,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=450,
        temperature=0.2,
    )

# ─────────────────────────────────────────────────────────────
# Plantilla para cargar las herramientas MCP una vez
# ─────────────────────────────────────────────────────────────
_assistant_template = Agent(
    role="Asistente de gimnasio",
    goal="Ayudar con equipos, asistencias y proveedores usando las herramientas.",
    backstory="Asistente conciso.",
    tools=[],
    mcps=[mcp_server],
    llm=llm,
    verbose=False,
)

_mcp_tools_cache = None


def _get_mcp_tools():
    """Carga las herramientas MCP filtradas una sola vez."""
    global _mcp_tools_cache
    if _mcp_tools_cache is None:
        try:
            _mcp_tools_cache = _assistant_template.get_mcp_tools(
                _assistant_template.mcps
            )
            print(f"✅ {len(_mcp_tools_cache)} herramientas MCP filtradas cargadas")

            # ── DEBUG: nombres reales de las tools cargadas ──
            print("🔍 Nombres reales de las herramientas cargadas:")
            for tool in _mcp_tools_cache:
                print(f"   - {tool.name}")

        except Exception as e:
            print(f"⚠️  No se pudieron cargar las herramientas del MCP ({e})")
            _mcp_tools_cache = []
    return _mcp_tools_cache


_get_mcp_tools()


AGENT_BACKSTORY = (
    "Asistente de gimnasio en español. SOLO respondes sobre equipos, "
    "asistencias y proveedores del gimnasio, usando las herramientas "
    "disponibles. Si la pregunta no tiene relación con esos tres temas "
    "(aunque sea sobre otro tema general), responde exactamente: "
    "\"Solo puedo ayudarte con equipos, asistencias y proveedores del "
    "gimnasio.\" y no uses ninguna herramienta.\n\n"
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
    """Crea un agente limpio con las herramientas filtradas."""
    return Agent(
        role="Asistente de gimnasio",
        goal=(
            "Responder de forma breve, completa y con formato consistente "
            "sobre equipos, asistencias y proveedores del gimnasio. "
            "Rechazar cualquier pregunta fuera de esos temas."
        ),
        backstory=AGENT_BACKSTORY,
        tools=list(_get_mcp_tools()),
        llm=llm,
        verbose=True,
        max_iter=5,                # margen para: buscar ID → actuar → responder
    )


# ─────────────────────────────────────────────────────────────
# Throttle para evitar agotar la cuota de Groq
# ─────────────────────────────────────────────────────────────
_last_request_time = 0.0
_min_interval = 3.0  # segundos entre requests (máx ~20 req/min)


# ─────────────────────────────────────────────────────────────
# Procesamiento
# ─────────────────────────────────────────────────────────────
def process_message(user_input: str) -> str:
    global _last_request_time

    # Throttle: garantiza un mínimo de separación entre requests
    elapsed = time.time() - _last_request_time
    if elapsed < _min_interval:
        wait = _min_interval - elapsed
        print(f"⏳ Throttle: esperando {wait:.1f}s")
        time.sleep(wait)

    try:
        agent = _create_agent()
        task = Task(
            description=user_input,
            expected_output=(
                "Si es tema fuera de equipos/asistencias/proveedores, el "
                "mensaje fijo de rechazo. Si es una lista, tabla Markdown. "
                "Si es un solo dato, una frase breve en español."
            ),
            agent=agent,
        )
        with _agent_lock:
            response = str(agent.execute_task(task))
        _last_request_time = time.time()
    except Exception as exc:
        response = f"Error al procesar la solicitud: {exc}"
        _last_request_time = time.time()

    return response


def chat():
    print("\n🤖 Chat con CrewAI + MCP (escribe 'salir' para terminar):")
    while True:
        user_input = input("Usuario: ").strip()
        if user_input.lower() in ["salir", "exit"]:
            print("Conversación finalizada.")
            break
        if not user_input:
            continue
        response = process_message(user_input)
        print(f"Asistente: {response}\n")


if __name__ == "__main__":
    chat()