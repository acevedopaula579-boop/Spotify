"""Contrato HTTP de entrada y salida del servicio (Pydantic)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.variables import FILA_EJEMPLO, GENEROS

Genero = Literal[tuple(GENEROS)]


class Cancion(BaseModel):
    """Características conocidas antes del lanzamiento.

    Cualquier campo adicional, por ejemplo reproducciones_sem1, popularidad o es_hit, se
    rechaza con 422: el contrato solo admite lo que existe en el momento de decidir.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [FILA_EJEMPLO]})

    bailabilidad: float = Field(..., ge=0, le=1)
    energia: float = Field(..., ge=0, le=1)
    valencia: float = Field(..., ge=0, le=1)
    acustica: float = Field(..., ge=0, le=1)
    tempo: float = Field(..., ge=40, le=250, description="pulsos por minuto")
    duracion_min: float = Field(..., ge=0.5, le=15, description="minutos")
    volumen_db: float = Field(..., ge=-60, le=5, description="decibelios")
    anio: int = Field(..., ge=1990, le=2035, description="año de lanzamiento")
    colaboracion: int = Field(..., ge=0, le=1, description="1 si es una colaboración")
    genero: Genero


class Prediccion(BaseModel):
    """Predicción con la decisión que sostiene. Los campos de la otra tarea llegan en null."""

    tarea: Literal["regresion", "clasificacion"]
    modelo: str = Field(..., description="nombre y fecha del modelo cargado")
    # regresión
    popularidad_esperada: float | None = None
    rango: list[float] | None = Field(None, description="popularidad esperada ± MAE del modelo")
    corte_promocion: float | None = None
    # clasificación
    probabilidad_hit: float | None = None
    umbral: float | None = None
    # decisión, en ambas tareas
    promocionar: bool
    explicacion: str


class Salud(BaseModel):
    estado: Literal["ok", "sin_modelo"]
    modelo_cargado: bool
    ruta_modelo: str
    modelo: str | None = None
    tarea: Literal["regresion", "clasificacion"] | None = None
