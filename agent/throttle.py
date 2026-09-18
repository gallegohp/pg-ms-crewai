"""Throttle simple por cuenta de Groq.

Cada agente usa una cuenta distinta con su propio límite de peticiones
por minuto, así que cada uno necesita su propio throttle en vez de
compartir un único contador global.
"""

import threading
import time


class Throttle:
    def __init__(self, min_interval: float = 3.0):
        self._min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Bloquea lo necesario para respetar la separación mínima entre requests."""
        with self._lock:
            elapsed = time.time() - self._last
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last = time.time()
