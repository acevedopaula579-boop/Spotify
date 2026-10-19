"""Datos, variables de entrada y preprocesamiento comunes a los dos sprints.

El servicio importa este módulo al cargar el modelo: si un pipeline usa una función que no
está aquí (por ejemplo, una variable derivada definida solo en el notebook), el archivo
.pkl no se puede cargar. Por eso las variables derivadas se definen en este módulo.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "canciones.csv"

# Partición temporal: se entrena con anio < ANIO_CORTE y se evalúa con anio >= ANIO_CORTE.
ANIO_CORTE = 2020

# Columnas que el servicio recibe en cada petición: las que existen antes del lanzamiento.
COLUMNAS_ENTRADA = [
    "bailabilidad",
    "energia",
    "valencia",
    "acustica",
    "tempo",
    "duracion_min",
    "volumen_db",
    "anio",
    "colaboracion",
    "genero",
]
CATEGORICAS = ["genero"]
GENEROS = ["pop", "urbano", "rock", "electronica", "indie"]

# Columnas que no existen en el momento de decidir o que se derivan del objetivo.
COLUMNAS_POSTERIORES = ["reproducciones_sem1", "popularidad", "es_hit"]

# Canción de ejemplo usada por las pruebas y la prueba de humo.
FILA_EJEMPLO = {
    "bailabilidad": 0.72,
    "energia": 0.65,
    "valencia": 0.55,
    "acustica": 0.12,
    "tempo": 118.0,
    "duracion_min": 3.4,
    "volumen_db": -6.5,
    "anio": 2026,
    "colaboracion": 1,
    "genero": "urbano",
}


COLUMNAS_DERIVADAS = [
    "duracion_c2",
    "valencia_acustica",
    "colab_indie",
    "pop_energia",
    "pop_bailabilidad",
    "urbano_bailabilidad",
]


def derivadas(X: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables derivadas. Se calculan fila a fila, sin aprender de otros registros."""
    X = X.copy()
    X["duracion_c2"] = (X["duracion_min"] - 3.5) ** 2
    X["valencia_acustica"] = X["valencia"] * X["acustica"]
    X["colab_indie"] = (X["colaboracion"] * (X["genero"] == "indie")).astype(float)
    X["pop_energia"] = ((X["genero"] == "pop").astype(float) * X["energia"])
    X["pop_bailabilidad"] = ((X["genero"] == "pop").astype(float) * X["bailabilidad"])
    X["urbano_bailabilidad"] = ((X["genero"] == "urbano").astype(float) * X["bailabilidad"])
    return X


def preprocesamiento() -> ColumnTransformer:
    """Codificación de género y escalado de las numéricas, aplicado después de `derivadas`."""
    numericas = [c for c in COLUMNAS_ENTRADA if c not in CATEGORICAS] + COLUMNAS_DERIVADAS
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
            ("num", StandardScaler(), numericas),
        ]
    )


def cargar_canciones(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV ordenado por año de lanzamiento."""
    return pd.read_csv(ruta).sort_values("anio", kind="stable").reset_index(drop=True)


def particion_temporal(datos: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Máscaras de entrenamiento (1995-2019) y prueba (2020-2025)."""
    entrenamiento = datos["anio"] < ANIO_CORTE
    return entrenamiento, ~entrenamiento


def resumen_periodo(datos: pd.DataFrame, mascara: pd.Series) -> dict:
    """Periodo y número de canciones de una partición, para la ficha."""
    anios = datos.loc[mascara, "anio"]
    return {"periodo": f"{anios.min()}-{anios.max()}", "n": int(mascara.sum())}
