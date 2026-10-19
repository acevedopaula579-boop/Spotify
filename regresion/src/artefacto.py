"""Guardar y cargar el modelo: models/modelo.pkl, el único archivo que cada estudiante reemplaza.

El archivo contiene un diccionario:

    {"pipeline": <Pipeline de scikit-learn entrenado>, "ficha": {...}, "columnas_entrada": [...]}

La ficha dice qué predice el modelo, con qué datos, qué tan bien y qué decisión sostiene.
`guardar` exige los campos mínimos y comprueba que el pipeline prediga antes de escribir.
"""

import json
import os
import platform
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.variables import COLUMNAS_ENTRADA, FILA_EJEMPLO, RAIZ

RUTA_MODELO = RAIZ / os.environ.get("RUTA_MODELO", "models/modelo.pkl")

CAMPOS_OBLIGATORIOS = [
    "nombre",
    "tarea",
    "objetivo",
    "momento_prediccion",
    "variables_excluidas",
    "linea_base",
    "desempeno",
    "decision",
]
TAMANO_MAXIMO_MB = 20  # el plan gratuito de Render tiene 512 MB de memoria


def validar_ficha(ficha: dict) -> None:
    """Lanza ValueError si la ficha no tiene lo que el servicio necesita."""
    faltan = [c for c in CAMPOS_OBLIGATORIOS if c not in ficha]
    if faltan:
        raise ValueError(f"La ficha no tiene los campos obligatorios: {faltan}")
    if ficha["tarea"] == "regresion":
        if "mae" not in ficha["desempeno"] or "corte_promocion" not in ficha["decision"]:
            raise ValueError("Una regresión necesita desempeno['mae'] y decision['corte_promocion']")
    elif ficha["tarea"] == "clasificacion":
        if "umbral" not in ficha["decision"]:
            raise ValueError("Una clasificación necesita decision['umbral']")
    else:
        raise ValueError("ficha['tarea'] debe ser 'regresion' o 'clasificacion'")


def guardar(pipeline, ficha: dict, ruta: Path = RUTA_MODELO) -> dict:
    """Valida la ficha, hace una predicción de prueba y escribe el archivo del modelo."""
    validar_ficha(ficha)
    if ficha["tarea"] == "clasificacion" and not hasattr(pipeline, "predict_proba"):
        raise ValueError("Un modelo de clasificación debe tener predict_proba")
    pipeline.predict(pd.DataFrame([FILA_EJEMPLO])[COLUMNAS_ENTRADA])

    ficha = {
        **ficha,
        "fecha": ficha.get("fecha", date.today().isoformat()),
        "versiones": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    modelo = {"pipeline": pipeline, "ficha": ficha, "columnas_entrada": list(COLUMNAS_ENTRADA)}

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta, compress=3)
    megas = ruta.stat().st_size / 1e6
    if megas > TAMANO_MAXIMO_MB:
        ruta.unlink()
        raise ValueError(f"El modelo pesa {megas:.1f} MB; el límite es {TAMANO_MAXIMO_MB} MB.")
    print(f"Modelo guardado en {ruta} ({megas:.2f} MB)")
    print(json.dumps(ficha, ensure_ascii=False, indent=2))
    return modelo


def cargar(ruta: Path = RUTA_MODELO) -> dict:
    """Carga el modelo y avisa si se creó con otra versión de scikit-learn."""
    modelo = joblib.load(ruta)
    for clave in ("pipeline", "ficha", "columnas_entrada"):
        if clave not in modelo:
            raise ValueError(f"El archivo no tiene la clave '{clave}'. Guárdelo con src.artefacto.guardar.")
    validar_ficha(modelo["ficha"])
    version = modelo["ficha"].get("versiones", {}).get("scikit_learn")
    if version != sklearn.__version__:
        print(f"ADVERTENCIA: el modelo se creó con scikit-learn {version} y el entorno tiene {sklearn.__version__}.")
    return modelo
