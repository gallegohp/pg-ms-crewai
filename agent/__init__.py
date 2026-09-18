"""
Módulo de Agente Conversacional — litellm directo, sin CrewAI.

Este __init__ prepara el entorno UNA sola vez (limpieza de variables de
Google y carga del .env) antes de que cualquier submódulo importe
litellm, para que no dude de qué proveedor usar.
"""

import os

for _var in (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
):
    os.environ.pop(_var, None)

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path), override=False)

# Segunda limpieza por si el .env las reintrodujo
for _var in (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
):
    os.environ.pop(_var, None)
