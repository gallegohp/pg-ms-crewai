"""
Agente Conversacional con CrewAI + MCP.
Optimizado para reducir el consumo de tokens del LLM.

Estrategia de reducción de tokens:
- El agente se recrea en cada petición, evitando que CrewAI acumule
  contexto interno en el AgentExecutor y dispare el payload del LLM.
- Las herramientas MCP se cargan una sola vez y se reutilizan.
- El system prompt del agente es mínimo.
"""

import os
import json
import threading
from dotenv import load_dotenv

_agent_lock = threading.Lock()

# Desactivar telemetría de CrewAI
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Cargar .env
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path))

from crewai import Agent, Task, LLM
from crewai.mcp import MCPServerSSE

# Aplicar parche de compatibilidad con Groq (cache_breakpoint).
# Con Gemini no es necesario, pero si en algún momento vuelves a Groq,
# ya lo tienes activo sin tocar nada más.
try:
    from agent.litellm_patch import apply_patch
    apply_patch()
except ImportError:
    pass

# -------------------------------------------------------------
# MCP Server
# -------------------------------------------------------------
mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
mcp_server = MCPServerSSE(url=mcp_server_url)

# -------------------------------------------------------------
# LLM
# -------------------------------------------------------------
# Detectamos el proveedor por el prefijo del modelo.
# - Si empieza por "gemini/", usamos GEMINI_API_KEY.
# - Si empieza por "groq/", usamos GROQ_API_KEY y base_url de Groq.
# - Fallback: OPENAI_API_KEY.
llm_model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")

if llm_model.startswith("gemini/"):
    api_key = os.getenv("GEMINI_API_KEY", "")
    llm = LLM(
        model=llm_model,
        api_key=api_key,
        max_tokens=512,
    )
elif llm_model.startswith("groq/") or llm_model.startswith("groq/groq/"):
    api_key = os.getenv("GROQ_API_KEY", "")
    llm = LLM(
        model=llm_model,
        api_key=api_key,
        max_tokens=512,
        base_url="https://api.groq.com/openai/v1",
    )
else:
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = LLM(
        model=llm_model,
        api_key=api_key,
        max_tokens=512,
    )

# -------------------------------------------------------------
# Plantilla (solo para cargar las herramientas MCP una vez)
# -------------------------------------------------------------
# Este agente NO se usa para ejecutar tareas. Solo existe para que
# CrewAI resuelva las herramientas del MCP. Las herramientas se
# cachean y se pasan a los agentes frescos que se crean por petición.
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
    """Carga las herramientas MCP una sola vez y las reutiliza."""
    global _mcp_tools_cache
    if _mcp_tools_cache is None:
        try:
            _mcp_tools_cache = _assistant_template.get_mcp_tools(
                _assistant_template.mcps
            )
            print(f"✅ {len(_mcp_tools_cache)} herramientas MCP cargadas")
        except Exception as e:
            print(f"⚠️  No se pudieron cargar las herramientas del MCP ({e})")
            _mcp_tools_cache = []
    return _mcp_tools_cache


# Cargar al arrancar para fallar rápido si el MCP no responde
_get_mcp_tools()


# -------------------------------------------------------------
# Fábrica de agentes (uno nuevo por petición)
# -------------------------------------------------------------
def _create_agent() -> Agent:
    """Crea un agente limpio, sin contexto acumulado de peticiones previas."""
    return Agent(
        role="Asistente de gimnasio",
        goal="Ayudar con equipos, asistencias y proveedores usando las herramientas.",
        backstory=(
            "Asistente conciso en español. "
            "Si el usuario menciona un equipo por NOMBRE y necesitas su ID para otra operación, "
            "PRIMERO usa `consultar_equipos` con el nombre para obtener el ID real. "
            "NUNCA inventes IDs. "
            "Usa exactamente los valores de enum indicados en las descripciones (mayúsculas incluidas)."
        ),
        tools=list(_get_mcp_tools()),
        llm=llm,
        verbose=True,
        max_iter=3,
    )


# -------------------------------------------------------------
# Historial
# -------------------------------------------------------------
HISTORY_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "history", "conversacion.json")
)


def get_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_message(role: str, content: str):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    history = get_history()
    history.append({"role": role, "content": content})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# -------------------------------------------------------------
# Procesamiento
# -------------------------------------------------------------
def process_message(user_input: str) -> str:
    save_message("user", user_input)

    try:
        agent = _create_agent()   # agente nuevo, contexto limpio
        task = Task(
            description=user_input,
            expected_output="Respuesta breve en español.",
            agent=agent,
        )
        with _agent_lock:
            response = str(agent.execute_task(task))
    except Exception as exc:
        response = f"Error al procesar la solicitud: {exc}"

    save_message("assistant", response)
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