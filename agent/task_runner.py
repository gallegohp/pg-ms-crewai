"""
Ejecuta una tarea de un agente con reintentos.

gpt-oss-120b en Groq tiene dos fallas conocidas (reportadas por otros
usuarios en la comunidad de Groq, más frecuentes con varias tools
disponibles a la vez):
1. A veces intenta invocar una tool interna que no existe en nuestra
   lista (ej. un buscador tipo "repo_browser.search") en vez de una
   tool real -> CrewAI lo revienta con "Invalid response from LLM
   call - None or empty" o un 400 de Groq.
2. A veces NO llama ninguna tool y responde directamente inventando
   el dato (ej. "3 equipos en mantenimiento" sin haber consultado
   nada) -> esto no lanza ningún error, así que hay que detectarlo
   a mano viendo si realmente se usó alguna tool.

Por eso esta función no solo reintenta ante excepciones: también
escucha el bus de eventos de CrewAI para contar tools usadas durante
la tarea, y si la respuesta no fue ni el rechazo fijo esperado ni el
resultado de usar una tool, la trata como sospechosa de estar
inventada y reintenta con un agente y una tarea nuevos.
"""

import threading
from typing import Callable

from crewai import Agent, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent


def run_with_retries(
    agent_factory: Callable[[], Agent],
    task_description: str,
    expected_output: str,
    lock: threading.Lock,
    rejection_message: str,
    retries: int = 2,
) -> str:
    last_exc: Exception | None = None
    rejection_message = rejection_message.strip()

    for _ in range(retries + 1):
        tool_calls: list[str] = []

        def _on_tool_finished(source, event, _sink=tool_calls) -> None:
            _sink.append(getattr(event, "tool_name", ""))

        crewai_event_bus.on(ToolUsageFinishedEvent)(_on_tool_finished)
        try:
            agent = agent_factory()
            task = Task(
                description=task_description,
                expected_output=expected_output,
                agent=agent,
            )
            with lock:
                result = str(agent.execute_task(task))
        except Exception as exc:
            last_exc = exc
            continue
        finally:
            crewai_event_bus.off(ToolUsageFinishedEvent, _on_tool_finished)

        if not tool_calls and result.strip() != rejection_message:
            last_exc = RuntimeError(
                f"Respuesta sin usar ninguna herramienta ni ser el rechazo "
                f"esperado (posible dato inventado): {result!r}"
            )
            continue

        return result

    return f"Error al procesar la solicitud: {last_exc}"
