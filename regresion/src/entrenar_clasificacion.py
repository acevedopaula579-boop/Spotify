"""Sprint 2, clasificación: entrena el clasificador de es_hit y escribe models/modelo.pkl.

Mismas variables de entrada y misma partición temporal que el sprint 1.
Entrenado con las canciones de 1995 a 2019 y evaluado con las de 2020 a 2025.
El umbral de promoción fijado por negocio es 0.27, optimizando el costo
asimétrico de los errores (FN=3, FP=1).

    python -m src.entrenar_clasificacion
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

NOMBRE = "clasificador logístico regularizado + interacciones"
UMBRAL = 0.27  # umbral fijo de promoción requerido para la toma de decisiones
COSTO_FALSO_NEGATIVO = 3  # dejar sin promoción una canción que sí fue hit
COSTO_FALSO_POSITIVO = 1  # promocionar una canción que no fue hit
VARIABLES_EXCLUIDAS = {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "popularidad": "define es_hit: usarla como entrada equivale a conocer la respuesta",
}


def construir_pipeline():
    """Debe recibir las COLUMNAS_ENTRADA crudas y tener predict_proba."""
    return make_pipeline(
        FunctionTransformer(derivadas),
        preprocesamiento(),
        LogisticRegression(
            C=3.0,
            class_weight={0: 1.0, 1: 1.25},
            max_iter=1000,
            random_state=42,
        ),
    )


def costo(real, pred) -> int:
    """Calcula el costo total de los errores según la matriz de costos de negocio."""
    falsos_negativos = int(((real == 1) & ~pred).sum())
    falsos_positivos = int(((real == 0) & pred).sum())
    return COSTO_FALSO_NEGATIVO * falsos_negativos + COSTO_FALSO_POSITIVO * falsos_positivos


def entrenar() -> dict:
    datos = cargar_canciones()
    entrenamiento, prueba = particion_temporal(datos)
    X, y = datos[COLUMNAS_ENTRADA], datos["es_hit"]

    pipeline = construir_pipeline().fit(X[entrenamiento], y[entrenamiento])
    prob = pipeline.predict_proba(X[prueba])[:, 1]
    pred = prob >= UMBRAL
    real = y[prueba].to_numpy()
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()

    ficha = {
        "nombre": NOMBRE,
        "tarea": "clasificacion",
        "objetivo": "Predecir si la canción será un hit (es_hit)",
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
            "costo": costo(real, pred),
            "roc_auc": round(float(roc_auc_score(real, prob)), 3),
            "brier_score": round(float(brier_score_loss(real, prob)), 3),
            "precision": round(float(precision_score(real, pred)), 3),
            "recall": round(float(recall_score(real, pred)), 3),
            "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "decision": {
            "umbral": UMBRAL,
            "costos": {"falso_negativo": COSTO_FALSO_NEGATIVO, "falso_positivo": COSTO_FALSO_POSITIVO},
            "regla": "promocionar si la probabilidad de hit supera el umbral",
            "nota": "los costos los fija quien responde por el presupuesto, no el modelo",
        },
        "limites": "describe asociaciones en datos sintéticos; probabilidades ajustadas por costos asimétricos",
    }
    return guardar(pipeline, ficha)


if __name__ == "__main__":
    entrenar()
