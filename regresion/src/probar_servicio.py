"""Prueba de humo contra el servicio, local o publicado en Render.

    python -m src.probar_servicio                                   # servicio local
    python -m src.probar_servicio https://su-servicio.onrender.com  # servicio publicado

El código de salida es el número de fallas.
"""

import sys
import time

import requests

from src.artefacto import CAMPOS_OBLIGATORIOS
from src.variables import FILA_EJEMPLO

url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
fallas = 0


def paso(nombre: str, ok: bool, detalle: str = "") -> None:
    global fallas
    fallas += 0 if ok else 1
    print(f"[{'OK' if ok else 'FALLA'}] {nombre} {detalle}".rstrip())


inicio = time.time()
try:
    r = requests.get(f"{url}/salud", timeout=90)  # el plan gratuito de Render tarda cerca de un minuto en despertar
except requests.RequestException as error:
    print(f"[FALLA] conexión ({error})")
    sys.exit(1)
paso("salud", r.status_code == 200 and r.json().get("modelo_cargado"), f"({time.time() - inicio:.0f} s)")

ficha = requests.get(f"{url}/modelo", timeout=30).json()
paso("ficha completa", all(c in ficha for c in CAMPOS_OBLIGATORIOS), f"{ficha.get('nombre')} ({ficha.get('tarea')})")

r = requests.post(f"{url}/predecir", json=FILA_EJEMPLO, timeout=30)
paso("predicción", r.status_code == 200, r.json().get("explicacion", r.text[:120]))

r = requests.post(f"{url}/predecir", json={**FILA_EJEMPLO, "reproducciones_sem1": 1000}, timeout=30)
paso("rechaza variables posteriores al lanzamiento", r.status_code == 422)

print("\nResultado:", "servicio en orden" if fallas == 0 else f"{fallas} falla(s)")
sys.exit(fallas)
