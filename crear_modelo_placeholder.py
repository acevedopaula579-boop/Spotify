"""Genera un modelo de mentira para la simulacion del servidor.
Reemplzar por el PKL de la regresion"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

BASE = Path(__file__).parent / "app" / "modelos"
columnas = json.loads((BASE / "config.json").read_text(encoding="utf-8"))["columnas"]

rng = np.random.default_rng(42)
X = rng.random((200, len(columnas)))
y = rng.random(200) * 100

joblib.dump(DummyRegressor(strategy="constant", constant=50.0).fit(X, y), BASE / "regresion_popularidad.pkl")
print("Modelo placeholder creado en", BASE / "regresion_popularidad.pkl")
