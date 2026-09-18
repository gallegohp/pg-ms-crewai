"""
Punto de entrada del agente conversacional.

Orquesta entre el agente operacional (equipos/asistencias/proveedores) y
el de analítica (ingresos/mora/afluencia) según el tema del mensaje, cada
uno con su propia cuenta de Groq. Ver agent/orchestrator.py,
agent/operational_agent.py y agent/analytics_agent.py.

`app.py` solo depende de `process_message`, así que esta capa de
orquestación es transparente para la API Flask.
"""

from agent.analytics_agent import process_analitica
from agent.operational_agent import process_operacional
from agent.orchestrator import RECHAZO, classify


def process_message(user_input: str) -> str:
    tema = classify(user_input)
    if tema == "operacional":
        return process_operacional(user_input)
    if tema == "analitica":
        return process_analitica(user_input)
    return RECHAZO


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
