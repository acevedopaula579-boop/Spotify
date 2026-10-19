"""Servicio de predicción: carga models/modelo.pkl al arrancar y expone la ficha y las predicciones.

    uvicorn src.app.main:app --reload --port 8000     # documentación en http://127.0.0.1:8000/docs

El servicio no tiene reglas propias: la tarea (regresión o clasificación), el corte o el umbral
y el error esperado se leen de la ficha del modelo. Cambiar de modelo es cambiar el .pkl.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from src.app.esquemas import Cancion, Prediccion, Salud
from src.artefacto import RUTA_MODELO, cargar

MAXIMO_LOTE = 1000


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carga el modelo una sola vez, al arrancar."""
    app.state.modelo = cargar(RUTA_MODELO) if RUTA_MODELO.exists() else None
    if app.state.modelo:
        ficha = app.state.modelo["ficha"]
        print(f"Modelo cargado: {ficha['nombre']} ({ficha['tarea']}, {ficha.get('fecha')})")
    else:
        print(f"No existe {RUTA_MODELO}")
    yield


app = FastAPI(title="Predicción de popularidad de canciones", version="1.0.0", lifespan=ciclo_de_vida)


def modelo_cargado(request: Request) -> dict:
    if request.app.state.modelo is None:
        raise HTTPException(status_code=503, detail=f"No hay modelo en {RUTA_MODELO.name}")
    return request.app.state.modelo


def etiqueta(ficha: dict) -> str:
    return f"{ficha['nombre']} ({ficha.get('fecha', 'sin fecha')})"


def predecir_filas(modelo: dict, canciones: list[Cancion]) -> list[Prediccion]:
    ficha, pipeline = modelo["ficha"], modelo["pipeline"]
    X = pd.DataFrame([c.model_dump() for c in canciones])[modelo["columnas_entrada"]]
    nombre = etiqueta(ficha)
    salida = []

    if ficha["tarea"] == "regresion":
        mae = float(ficha["desempeno"]["mae"])
        corte = float(ficha["decision"]["corte_promocion"])
        for valor in map(float, pipeline.predict(X)):
            promocionar = valor >= corte
            salida.append(
                Prediccion(
                    tarea="regresion",
                    modelo=nombre,
                    popularidad_esperada=round(valor, 1),
                    rango=[round(valor - mae, 1), round(valor + mae, 1)],
                    corte_promocion=corte,
                    promocionar=promocionar,
                    explicacion=f"popularidad esperada {valor:.1f} (±{mae:g}) "
                    f"{'supera' if promocionar else 'no supera'} el corte {corte:g}",
                )
            )
    else:
        umbral = float(ficha["decision"]["umbral"])
        for prob in map(float, pipeline.predict_proba(X)[:, 1]):
            promocionar = prob >= umbral
            salida.append(
                Prediccion(
                    tarea="clasificacion",
                    modelo=nombre,
                    probabilidad_hit=round(prob, 3),
                    umbral=umbral,
                    promocionar=promocionar,
                    explicacion=f"probabilidad de hit {prob:.2f} "
                    f"{'supera' if promocionar else 'no supera'} el umbral {umbral:g}",
                )
            )
    return salida


@app.get("/", include_in_schema=False)
def raiz() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/salud", response_model=Salud)
def salud(request: Request) -> Salud:
    modelo = request.app.state.modelo
    return Salud(
        estado="ok" if modelo else "sin_modelo",
        modelo_cargado=modelo is not None,
        ruta_modelo=RUTA_MODELO.name,
        modelo=etiqueta(modelo["ficha"]) if modelo else None,
        tarea=modelo["ficha"]["tarea"] if modelo else None,
    )


@app.get("/modelo")
def ficha(request: Request) -> dict:
    """Ficha del modelo: qué predice, con qué datos, qué tan bien y qué decisión sostiene."""
    return modelo_cargado(request)["ficha"]


@app.post("/predecir", response_model=Prediccion)
def predecir(cancion: Cancion, request: Request) -> Prediccion:
    """Predicción y decisión para una canción."""
    return predecir_filas(modelo_cargado(request), [cancion])[0]


@app.post("/predecir_lote", response_model=list[Prediccion])
def predecir_lote(canciones: list[Cancion], request: Request) -> list[Prediccion]:
    """Predicción y decisión para hasta 1.000 canciones."""
    if len(canciones) > MAXIMO_LOTE:
        raise HTTPException(status_code=413, detail=f"Máximo {MAXIMO_LOTE} canciones por petición")
    return predecir_filas(modelo_cargado(request), canciones)
