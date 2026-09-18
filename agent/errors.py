"""Excepciones propias, para que app.py distinga qué código HTTP devolver."""


class RateLimitExceeded(Exception):
    """El proveedor del LLM (Groq) devolvió un rate limit."""
