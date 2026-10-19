import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

RAIZ = Path(__file__).resolve().parent.parent
BASE = RAIZ / "app" / "modelos"
REGRESION = RAIZ / "regresion"
RUTA_DATOS = REGRESION / "data" / "canciones.csv"
RUTA_UI = RAIZ / "app" / "static" / "index.html"

# El .pkl guarda referencias a src.variables.derivadas: `src` vive dentro de regresion/.
if str(REGRESION) not in sys.path:
    sys.path.insert(0, str(REGRESION))

app = FastAPI(
    title="API de prediccion de popularidad de canciones",
    description="Sirve un modelo de regresion (PKL) que predice el indice de popularidad.",
)

config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))

# Carga una sola vez, al arrancar el proceso (no en cada request).
artefacto = joblib.load(BASE / config["archivo_modelo"])
modelo = artefacto["pipeline"]
COLUMNAS: list[str] = list(artefacto["columnas_entrada"])
FICHA: dict = artefacto["ficha"]

MARGEN_RANGO = 0.10  # 10% del rango observado, para no rechazar valores reales


def rangos_desde_datos(columnas: list[str]) -> dict[str, tuple[float, float]]:
    """Minimo y maximo de cada columna numerica en canciones.csv, con margen."""
    if not RUTA_DATOS.exists():
        print(f"Aviso: no existe {RUTA_DATOS}; no se validan rangos.")
        return {}
    datos = pd.read_csv(RUTA_DATOS)
    rangos: dict[str, tuple[float, float]] = {}
    for columna in columnas:
        if columna not in datos.columns:
            continue
        serie = pd.to_numeric(datos[columna], errors="coerce").dropna()
        if serie.empty:
            continue
        minimo, maximo = float(serie.min()), float(serie.max())
        margen = (maximo - minimo) * MARGEN_RANGO
        rangos[columna] = (minimo - margen, maximo + margen)
    return rangos


def categorias_de_genero(pipeline) -> list[str]:
    """Categorias que el OneHotEncoder del pipeline aprendio para `genero`."""
    pendientes = [pipeline]
    while pendientes:
        objeto = pendientes.pop()
        if hasattr(objeto, "categories_") and hasattr(objeto, "drop_idx_"):
            return [str(c) for c in objeto.categories_[0]]
        for atributo in ("steps", "transformers_"):
            for elemento in getattr(objeto, atributo, None) or []:
                pendientes.append(elemento[1])
    return []


RANGOS = rangos_desde_datos([c for c in COLUMNAS if c != "genero"])
GENEROS = categorias_de_genero(modelo)
if not GENEROS:
    print("Aviso: el pipeline no expone categorias de `genero`; no se valida ese campo.")

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
                {"m": FICHA["nombre"], "v": config["version"],
                 "e": json.dumps(entrada), "p": valor},
            )
    except Exception as e:
        print("Aviso: no se pudo guardar la prediccion:", e)


class Entrada(BaseModel):
    """Caracteristicas conocidas antes del lanzamiento."""

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
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
        },
    }

    bailabilidad: float
    energia: float
    valencia: float
    acustica: float
    tempo: float
    duracion_min: float
    volumen_db: float
    anio: float
    colaboracion: float
    genero: str

    @field_validator("genero")
    @classmethod
    def genero_conocido(cls, valor: str) -> str:
        if GENEROS and valor not in GENEROS:
            raise ValueError(f"genero desconocido: {valor!r}; use uno de {GENEROS}")
        return valor

    @model_validator(mode="after")
    def dentro_de_rangos(self):
        fuera = []
        for nombre, (minimo, maximo) in RANGOS.items():
            valor = getattr(self, nombre, None)
            if isinstance(valor, (int, float)) and not minimo <= valor <= maximo:
                fuera.append(f"{nombre}={valor:g} (permitido {minimo:g} a {maximo:g})")
        if fuera:
            raise ValueError("Valores fuera de rango: " + "; ".join(fuera))
        return self


_faltan = set(COLUMNAS) - set(Entrada.model_fields)
_sobran = set(Entrada.model_fields) - set(COLUMNAS)
if _faltan or _sobran:
    raise RuntimeError(
        f"El esquema no coincide con columnas_entrada del .pkl: "
        f"faltan={sorted(_faltan)} sobran={sorted(_sobran)}"
    )


# endpoints
@app.get("/")
def raiz():
    return {"servicio": "popularidad-canciones", "docs": "/docs", "ui": "/ui", "estado": "ok"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    if not RUTA_UI.exists():
        raise HTTPException(status_code=404, detail="UI no disponible")
    return HTMLResponse(RUTA_UI.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"estado": "ok", "base_de_datos": engine is not None}


@app.get("/modelo/info")
def info():
    return {
        "nombre": FICHA["nombre"],
        "version": config["version"],
        "columnas": COLUMNAS,
        "generos": GENEROS,
        "tipo": type(modelo).__name__,
    }


@app.post("/predict/popularidad")
def predecir(entrada: Entrada):
    fila = pd.DataFrame([entrada.model_dump()], columns=COLUMNAS)
    valor = float(modelo.predict(fila)[0])
    guardar_prediccion(entrada.model_dump(), valor)
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
