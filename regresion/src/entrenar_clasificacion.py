"""Sprint 2, clasificación: entrena el clasificador de es_hit y escribe models/modelo.pkl.

Mismas variables de entrada y misma partición temporal que el sprint 1. El umbral de decisión
se elige minimizando el costo de los errores con probabilidades fuera de muestra dentro del
periodo de entrenamiento, de modo que 2020-2025 no interviene en ninguna decisión.

    python -m src.entrenar_clasificacion

Para usar su propio modelo, cambie `construir_pipeline`, `NOMBRE` y, si lo decide, los costos.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.artefacto import guardar
from src.variables import (
    COLUMNAS_ENTRADA,
    cargar_canciones,
    derivadas,
    particion_temporal,
    preprocesamiento,
    resumen_periodo,
)

NOMBRE = "logística + umbral por costos"
COSTO_FALSO_NEGATIVO = 3  # dejar sin promoción una canción que sí fue hit
COSTO_FALSO_POSITIVO = 1  # promocionar una canción que no fue hit
VARIABLES_EXCLUIDAS = {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "popularidad": "define es_hit: usarla como entrada equivale a conocer la respuesta",
}


def construir_pipeline():
    """Debe recibir las COLUMNAS_ENTRADA crudas y tener predict_proba."""
    return make_pipeline(FunctionTransformer(derivadas), preprocesamiento(), LogisticRegression(max_iter=1000))


def costo(real, pred) -> int:
    falsos_negativos = int(((real == 1) & ~pred).sum())
    falsos_positivos = int(((real == 0) & pred).sum())
    return COSTO_FALSO_NEGATIVO * falsos_negativos + COSTO_FALSO_POSITIVO * falsos_positivos


def elegir_umbral(X, y) -> float:
    """Umbral de menor costo sobre probabilidades fuera de muestra (ventana creciente)."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    reales, probs = [], []
    for ajuste, validacion in TimeSeriesSplit(n_splits=4).split(X):
        modelo = construir_pipeline().fit(X.iloc[ajuste], y.iloc[ajuste])
        probs.append(modelo.predict_proba(X.iloc[validacion])[:, 1])
        reales.append(y.iloc[validacion].to_numpy())
    real, prob = np.concatenate(reales), np.concatenate(probs)
    umbrales = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costos = [costo(real, prob >= u) for u in umbrales]
    return float(umbrales[int(np.argmin(costos))])


def entrenar() -> dict:
    datos = cargar_canciones()
    entrenamiento, prueba = particion_temporal(datos)
    X, y = datos[COLUMNAS_ENTRADA], datos["es_hit"]

    umbral = elegir_umbral(X[entrenamiento], y[entrenamiento])
    pipeline = construir_pipeline().fit(X[entrenamiento], y[entrenamiento])
    pred = pipeline.predict_proba(X[prueba])[:, 1] >= umbral
    real = y[prueba].to_numpy()
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()

    ficha = {
        "nombre": NOMBRE,
        "tarea": "clasificacion",
        "objetivo": "es_hit: 1 si la canción alcanza 70 puntos de popularidad",
        "momento_prediccion": "antes del lanzamiento: la disquera decide cuánto invertir en promoción",
        "variables_entrada": list(COLUMNAS_ENTRADA),
        "variables_excluidas": VARIABLES_EXCLUIDAS,
        "entrenamiento": resumen_periodo(datos, entrenamiento),
        "prueba": {**resumen_periodo(datos, prueba), "tipo": "temporal"},
        "linea_base": {
            "descripcion": "predecir siempre 'no hit'",
            "costo": costo(real, np.zeros_like(pred)),
        },
        "desempeno": {
            "precision": round(float(precision_score(real, pred)), 3),
            "recall": round(float(recall_score(real, pred)), 3),
            "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "costo": costo(real, pred),
        },
        "decision": {
            "umbral": umbral,
            "costos": {"falso_negativo": COSTO_FALSO_NEGATIVO, "falso_positivo": COSTO_FALSO_POSITIVO},
            "regla": "promocionar si la probabilidad de hit supera el umbral",
            "nota": "los costos los fija quien responde por el presupuesto, no el modelo",
        },
        "limites": "probabilidades sin calibrar; describe asociaciones en datos sintéticos",
    }
    return guardar(pipeline, ficha)


if __name__ == "__main__":
    entrenar()
