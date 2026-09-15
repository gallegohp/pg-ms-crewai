# agent/litellm_patch.py
import litellm

_original_completion = litellm.completion

def _strip_key_from_messages(messages, key):
    """Elimina una clave de cada mensaje en la lista, incluyendo content blocks."""
    if not isinstance(messages, list):
        return messages
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # Nivel del mensaje (ej: {"role": "system", "content": "...", "cache_breakpoint": {...}})
        msg.pop(key, None)
        # Nivel del bloque de contenido (ej: {"type": "text", "cache_breakpoint": {...}})
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop(key, None)
    return messages


def _patched_completion(*args, **kwargs):
    # 1. Eliminar la clave si viene como argumento de nivel superior
    kwargs.pop("cache_breakpoint", None)
    kwargs.pop("is_litellm", None)

    # 2. Eliminar la clave si está dentro de los mensajes
    if "messages" in kwargs:
        _strip_key_from_messages(kwargs["messages"], "cache_breakpoint")
        _strip_key_from_messages(kwargs["messages"], "is_litellm")

    return _original_completion(*args, **kwargs)


def apply_patch():
    """Aplica el parche a litellm.completion. Llamar una vez al inicio."""
    litellm.completion = _patched_completion