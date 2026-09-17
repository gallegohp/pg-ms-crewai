"""
Servidor Flask - API REST del Agente Conversacional CrewAI + MCP.
"""

import json
import os
import time
import threading
from collections import deque
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from agent.conversational_agent import process_message
from agent.llm_config import LLM_ANALITICA, LLM_ORQUESTADOR

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

HISTORY_PATH = Path(__file__).parent / "data" / "history" / "live_session.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
_history_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# Rate limit por minuto para /api/chat.
# Corta rápido (HTTP 429) antes de encolar trabajo al LLM, en vez de
# dejar que las solicitudes se acumulen esperando el throttle interno
# del agente. Ventana deslizante en memoria (un solo proceso/contenedor).
# ─────────────────────────────────────────────────────────────
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "15"))
_rate_lock = threading.Lock()
_request_times: deque = deque()


def _check_rate_limit() -> float:
    """Devuelve 0 si la solicitud está permitida, o los segundos a esperar."""
    now = time.time()
    with _rate_lock:
        while _request_times and now - _request_times[0] > 60:
            _request_times.popleft()
        if len(_request_times) >= RATE_LIMIT_PER_MINUTE:
            return round(60 - (now - _request_times[0]), 1)
        _request_times.append(now)
        return 0


def _load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _append_history(role: str, content: str) -> None:
    with _history_lock:
        messages = _load_history()
        messages.append({"role": role, "content": content})
        HISTORY_PATH.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/api/status", methods=["GET"])
def status():
    """Estado del servidor MCP y LLM."""
    return jsonify({
        "status": "online",
        "mcp": {
            "online": True,
            "mode": os.getenv("MCP_TRANSPORT", "sse"),
            "url": os.getenv("MCP_SERVER_URL", ""),
        },
        "llm": {"model": os.getenv("LLM_MODEL", "groq/openai/gpt-oss-120b")},
        "rate_limit": {"per_minute": RATE_LIMIT_PER_MINUTE},
        "agents": {
            "operacional": True,
            "analitica": LLM_ANALITICA is not None,
            "orquestador_con_llm": LLM_ORQUESTADOR is not None,
        },
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint que ejecuta un turno en el bucle del agente."""
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Mensaje requerido"}), 400

    wait = _check_rate_limit()
    if wait > 0:
        response = jsonify({
            "error": "Demasiadas solicitudes. Intenta de nuevo en unos segundos.",
            "retry_after": wait,
        })
        response.headers["Retry-After"] = str(int(wait) + 1)
        return response, 429

    _append_history("user", message)
    response = process_message(message)
    _append_history("assistant", response)
    return jsonify({
        "success": True,
        "response": response
    })


@app.route("/api/history", methods=["GET"])
def history():
    """Historial de la conversación (persistido en disco)."""
    return jsonify({"messages": _load_history()})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """Borra el historial de conversación."""
    with _history_lock:
        HISTORY_PATH.write_text("[]", encoding="utf-8")
    return jsonify({"success": True})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    print(f"🚀 Servidor Flask API en http://{host}:{port}")
    app.run(host=host, port=port, debug=False)