"""
Loop manual de function-calling — reemplaza el executor de CrewAI.

Llama al LLM con las tools disponibles; si pide una, la ejecuta contra
el MCP y le devuelve el resultado; repite hasta que responda en texto
plano. Si el modelo responde sin haber usado ninguna tool y sin ser el
rechazo fijo esperado, se trata como sospechoso de estar inventado y
se reintenta la conversación completa (hasta `retries` veces) antes de
devolver un error honesto.

Esto reemplaza tanto el Agent/Task de CrewAI como el detector de
"respuesta inventada" que antes dependía del bus de eventos de CrewAI
(agent/task_runner.py) — acá se sabe directamente si se usó una tool
porque el loop la ejecuta él mismo.
"""

import json

from agent.llm_client import chat
from agent.mcp_client import call_tool


def _run_once(api_key: str, system_prompt: str, user_input: str, tools: list, max_iters: int):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    used_tool = False

    for _ in range(max_iters):
        resp = chat(api_key, messages, tools=tools)
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls

        if not tool_calls:
            return (msg.content or "").strip(), used_tool

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            used_tool = True
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = call_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "", used_tool


def run(
    api_key: str,
    system_prompt: str,
    user_input: str,
    tools: list,
    rejection_message: str,
    retries: int = 2,
    max_iters: int = 5,
) -> str:
    rejection_message = rejection_message.strip()
    last_error = "Error al procesar la solicitud: sin respuesta del modelo."

    for _ in range(retries + 1):
        try:
            content, used_tool = _run_once(api_key, system_prompt, user_input, tools, max_iters)
        except Exception as exc:
            last_error = f"Error al procesar la solicitud: {exc}"
            continue

        if not content:
            last_error = "Error al procesar la solicitud: el modelo no devolvió respuesta."
            continue

        if not used_tool and content != rejection_message:
            last_error = (
                "Error al procesar la solicitud: respuesta sin usar ninguna "
                f"herramienta ni ser el rechazo esperado (posible dato "
                f"inventado): {content!r}"
            )
            continue

        return content

    return last_error
