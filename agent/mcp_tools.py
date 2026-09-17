"""
Carga las herramientas MCP UNA sola vez (una sola conexión SSE al
servidor MCP) y las reparte por agente según el nombre.

Solo se cargan las herramientas relevantes para reducir el esquema JSON
que CrewAI envía al LLM en cada request; cada agente además solo recibe
su propio subconjunto, así el esquema de un agente nunca crece por
tools que solo usa el otro.
"""

import os

from crewai import Agent
from crewai.mcp import MCPServerSSE
from crewai.mcp.filters import create_static_tool_filter

from agent.llm_config import LLM_OPERACIONAL

TOOLS_OPERACIONAL = [
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

TOOLS_ANALITICA = [
    # Ingresos (dinero)
    "reporte_ingresos_diarios",
    "reporte_ingresos_mensuales",
    "reporte_ingresos_por_membresia",
    "reporte_ingresos_ultimos_seis_meses",
    # Mora
    "reporte_mora",
    # Afluencia (entradas físicas, no dinero)
    "reporte_afluencia_hoy",
    "reporte_afluencia_por_dia",
]

_ALL_TOOL_NAMES = TOOLS_OPERACIONAL + TOOLS_ANALITICA

mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

mcp_server = MCPServerSSE(
    url=mcp_server_url,
    tool_filter=create_static_tool_filter(allowed_tool_names=_ALL_TOOL_NAMES),
    cache_tools_list=True,
)

# Agente "molde", solo para pedirle a CrewAI la lista de tools MCP.
# Nunca ejecuta una tarea real, por eso el LLM que usa es irrelevante.
_loader_template = Agent(
    role="loader",
    goal="Cargar herramientas MCP.",
    backstory="No se usa para responder.",
    tools=[],
    mcps=[mcp_server],
    llm=LLM_OPERACIONAL,
    verbose=False,
)

_tools_cache = None


def _load_all():
    global _tools_cache
    if _tools_cache is None:
        try:
            _tools_cache = _loader_template.get_mcp_tools(_loader_template.mcps)
            print(f"✅ {len(_tools_cache)} herramientas MCP cargadas (operacional + analítica)")
            print("🔍 Nombres reales de las herramientas cargadas:")
            for tool in _tools_cache:
                print(f"   - {tool.name}")
        except Exception as e:
            print(f"⚠️  No se pudieron cargar las herramientas del MCP ({e})")
            _tools_cache = []
    return _tools_cache


def tools_for(names: list) -> list:
    """Devuelve solo las tools cuyo nombre está en `names`."""
    wanted = set(names)
    return [t for t in _load_all() if t.name in wanted]


# Carga eager al importar el módulo, igual que antes.
_load_all()
