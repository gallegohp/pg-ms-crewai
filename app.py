"""
Servidor Flask - API REST del Agente Conversacional CrewAI + MCP.
"""

import json
import os
import threading
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from agent.conversational_agent import process_message

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

HISTORY_PATH = Path(__file__).parent / "data" / "history" / "live_session.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
_history_lock = threading.Lock()


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
        "llm": {"model": os.getenv("LLM_MODEL", "groq/openai/gpt-oss-120b")}
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint que ejecuta un turno en el bucle del agente."""
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Mensaje requerido"}), 400

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