"""
Servidor Flask - API REST del Agente Conversacional CrewAI + MCP.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from agent.conversational_agent import process_message

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


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
        "llm": {"model": os.getenv("LLM_MODEL", "groq/compound")}
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint que ejecuta un turno en el bucle del agente."""
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Mensaje requerido"}), 400

    response = process_message(message)
    return jsonify({
        "success": True,
        "response": response
    })


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    print(f"🚀 Servidor Flask API en http://{host}:{port}")
    app.run(host=host, port=port, debug=False)