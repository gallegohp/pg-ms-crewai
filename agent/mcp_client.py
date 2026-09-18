"""
Cliente MCP directo contra pg-ms-ai-new — sin CrewAI de por medio.

Se abre una conexión SSE nueva por cada llamada (listar tools al
arrancar, y una por cada tool call durante una conversación). No hay
sesión persistente: es más simple y evita depender de un event loop
compartido entre Flask (síncrono) y el cliente MCP (asíncrono), que era
parte de lo frágil de la integración de CrewAI.
"""

import asyncio
import json
import os
import time

from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

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

_ALL_TOOL_NAMES = set(TOOLS_OPERACIONAL) | set(TOOLS_ANALITICA)

_schema_cache: dict | None = None


async def _list_tools_async() -> dict:
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return {
                t.name: {"description": t.description or "", "inputSchema": t.inputSchema}
                for t in result.tools
                if t.name in _ALL_TOOL_NAMES
            }


def _load_schemas(retries: int = 10, delay: float = 3.0) -> dict:
    """Carga los schemas al arrancar, con reintentos.

    docker-compose `depends_on` solo espera a que el contenedor de
    pg-ms-ai-new arranque, no a que su servidor MCP esté escuchando
    de verdad — sin reintento acá, una carrera de arranque deja el
    caché vacío para siempre.
    """
    global _schema_cache
    if _schema_cache is None:
        last_exc = None
        for intento in range(1, retries + 1):
            try:
                _schema_cache = asyncio.run(_list_tools_async())
                print(f"✅ {len(_schema_cache)} herramientas MCP cargadas (operacional + analítica)")
                for name in _schema_cache:
                    print(f"   - {name}")
                return _schema_cache
            except Exception as e:
                last_exc = e
                print(f"⚠️  Intento {intento}/{retries} de cargar tools MCP falló ({e})")
                if intento < retries:
                    time.sleep(delay)
        print(f"⚠️  No se pudieron cargar las herramientas del MCP tras {retries} intentos ({last_exc})")
        _schema_cache = {}
    return _schema_cache


def get_tool_schemas(names: list) -> list:
    """Devuelve las tools pedidas en formato OpenAI function-calling."""
    schemas = _load_schemas()
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": schemas[name]["description"],
                "parameters": schemas[name]["inputSchema"],
            },
        }
        for name in names
        if name in schemas
    ]


async def _call_tool_async(name: str, arguments: dict) -> str:
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            texts = [c.text for c in result.content if hasattr(c, "text")]
            output = "\n".join(texts) if texts else json.dumps(result.structuredContent)
            if result.isError:
                return f"Error de la herramienta: {output}"
            return output


def call_tool(name: str, arguments: dict) -> str:
    try:
        return asyncio.run(_call_tool_async(name, arguments))
    except Exception as exc:
        return f"Error ejecutando la herramienta '{name}': {exc}"


# Carga eager al importar el módulo, igual que antes con CrewAI.
_load_schemas()
