import json
import os
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

BASE = Path(__file__).parent / "modelos"

app = FastAPI(
    title="API de prediccion de popularidad de canciones",
    description="Sirve un modelo de regresion (PKL) que predice el indice de popularidad.",
)

config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
COLUMNAS = config["columnas"]
modelo = joblib.load(BASE / config["archivo_modelo"])

# base de datos
engine = None
_url = os.getenv("DATABASE_URL")
if _url:
    try:
        from sqlalchemy import create_engine, text

        if _url.startswith("postgres://"):
            _url = _url.replace("postgres://", "postgresql://", 1)
        engine = create_engine(_url, pool_pre_ping=True)
        with engine.begin() as con:
            con.execute(text("""
                CREATE TABLE IF NOT EXISTS predicciones (
                    id SERIAL PRIMARY KEY,
                    modelo TEXT NOT NULL,
                    version TEXT NOT NULL,
                    entrada JSONB NOT NULL,
                    indice_popularidad FLOAT NOT NULL,
                    creado_en TIMESTAMP DEFAULT NOW()
                )
            """))
    except Exception as e:  # el servicio sigue funcionando sin BD
        print("Aviso: no se pudo conectar a la base de datos:", e)
        engine = None


def guardar_prediccion(entrada: dict, valor: float) -> None:
    if engine is None:
        return
    try:
        from sqlalchemy import text

        with engine.begin() as con:
            con.execute(
                text("INSERT INTO predicciones (modelo, version, entrada, indice_popularidad) "
                     "VALUES (:m, :v, CAST(:e AS JSONB), :p)"),
                {"m": config["nombre_modelo"], "v": config["version"],
                 "e": json.dumps(entrada), "p": valor},
            )
    except Exception as e:
        print("Aviso: no se pudo guardar la prediccion:", e)


RANGOS = {
    "duration_ms": (1, 3_600_000),
    "explicit": (0, 1),
    "danceability": (0, 1),
    "energy": (0, 1),
    "key": (-1, 11),
    "loudness": (-60, 5),
    "mode": (0, 1),
    "speechiness": (0, 1),
    "acousticness": (0, 1),
    "instrumentalness": (0, 1),
    "liveness": (0, 1),
    "valence": (0, 1),
    "tempo": (1, 300),
    "time_signature": (0, 7),
}


class Entrada(BaseModel):
    features: Dict[str, float]

    @field_validator("features")
    @classmethod
    def validar_rangos(cls, v):
        fuera = []
        for nombre, (minimo, maximo) in RANGOS.items():
            if nombre in v and not (minimo <= v[nombre] <= maximo):
                fuera.append(f"{nombre}={v[nombre]} (permitido {minimo} a {maximo})")
        if fuera:
            raise ValueError("Valores fuera de rango: " + "; ".join(fuera))
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "duration_ms": 210000, "explicit": 0, "danceability": 0.7,
                    "energy": 0.8, "key": 5, "loudness": -5.0, "mode": 1,
                    "speechiness": 0.05, "acousticness": 0.1,
                    "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.6,
                    "tempo": 120.0, "time_signature": 4,
                }
            }
        }
    }


# endpoints
@app.get("/")
def raiz():
    return {"servicio": "popularidad-canciones", "docs": "/docs", "estado": "ok"}


@app.get("/health")
def health():
    return {"estado": "ok", "base_de_datos": engine is not None}


@app.get("/modelo/info")
def info():
    return {
        "nombre": config["nombre_modelo"],
        "version": config["version"],
        "columnas": COLUMNAS,
        "tipo": type(modelo).__name__,
    }


@app.post("/predict/popularidad")
def predecir(entrada: Entrada):
    faltan = [c for c in COLUMNAS if c not in entrada.features]
    if faltan:
        raise HTTPException(status_code=422, detail=f"Faltan columnas: {faltan}")

    df = pd.DataFrame([[entrada.features[c] for c in COLUMNAS]], columns=COLUMNAS)
    valor = float(modelo.predict(df)[0])
    guardar_prediccion(entrada.features, valor)
    return {"indice_popularidad": round(valor, 2), "version_modelo": config["version"]}


@app.get("/metricas/regresion")
def metricas_regresion():
    return json.loads((BASE / "metricas_regresion.json").read_text(encoding="utf-8"))

@app.get("/predicciones/ultimas")
def ultimas_predicciones(limite: int = 10):
    if engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
    from sqlalchemy import text

    with engine.begin() as con:
        filas = con.execute(
            text("SELECT id, modelo, version, entrada, indice_popularidad, creado_en "
                 "FROM predicciones ORDER BY id DESC LIMIT :n"),
            {"n": min(max(limite, 1), 100)},
        ).mappings().all()
    return [dict(f) for f in filas]
