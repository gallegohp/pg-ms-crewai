"""
Agente Conversacional con CrewAI + MCP.
Optimizado para reducir el consumo de tokens del LLM.

Migrado a groq/compound (70K TPM vs 8K TPM de gpt-oss-120b).
Herramientas MCP filtradas para reducir el esquema enviado al LLM.
"""

import os
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
#
# 17 tools completas ≈ 4.000 tokens de esquema
# 5 tools filtradas ≈ 1.000 tokens de esquema
# Ahorro: ~75% menos tokens de entrada por request.

TOOLS_RELEVANTES = [
    # Equipos
    "consultar_equipos",
    "contar_equipos_por_estado",
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
    cache_tools_list=True,  # Reutiliza la lista entre requests
)

# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────
llm_model = os.getenv("LLM_MODEL", "groq/compound")
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
        raise RuntimeError(
            "GROQ_API_KEY está vacía. Revisa tu .env."
        )
    llm = LLM(
        model=llm_model,                          # "groq/compound"
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        max_tokens=512,
        max_retries=5,
    )
elif llm_model.startswith("gemini/"):
    llm = LLM(
        model=llm_model,
        api_key=os.getenv("GEMINI_API_KEY", ""),
        max_tokens=512,
    )
else:
    llm = LLM(
        model=llm_model,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=512,
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
        except Exception as e:
            print(f"⚠️  No se pudieron cargar las herramientas del MCP ({e})")
            _mcp_tools_cache = []
    return _mcp_tools_cache


_get_mcp_tools()


def _create_agent() -> Agent:
    """Crea un agente limpio con las herramientas filtradas."""
    return Agent(
        role="Asistente de gimnasio",
        goal="Ayudar con equipos, asistencias y proveedores usando las herramientas.",
        backstory=(
            "Asistente conciso en español. "
            "Si el usuario menciona un equipo por NOMBRE y necesitas su ID, "
            "PRIMERO usa `consultar_equipos` con el nombre para obtener el ID real. "
            "NUNCA inventes IDs. "
            "Usa exactamente los valores de enum indicados en las descripciones."
        ),
        tools=list(_get_mcp_tools()),
        llm=llm,
        verbose=True,
        max_iter=3,
    )


# ─────────────────────────────────────────────────────────────
# Procesamiento
# ─────────────────────────────────────────────────────────────
def process_message(user_input: str) -> str:
    try:
        agent = _create_agent()
        task = Task(
            description=user_input,
            expected_output="Respuesta breve en español.",
            agent=agent,
        )
        with _agent_lock:
            response = str(agent.execute_task(task))
    except Exception as exc:
        response = f"Error al procesar la solicitud: {exc}"

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