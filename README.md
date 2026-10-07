# API de popularidad de canciones

Servidor FastAPI que sirve un modelo de regresion (PKL) y, si hay `DATABASE_URL`,
guarda cada prediccion en PostgreSQL.

## Probar local (opcional)
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python crear_modelo_placeholder.py                # solo si no hay PKL
uvicorn app.main:app --reload
# http://localhost:8000/docs
```

## Desplegar en Render
1. Subir esta carpeta a un repo de GitHub.
2. Render -> New -> Blueprint -> elegir el repo (lee `render.yaml`: crea API + Postgres).
3. Cuando termine: `https://<tu-servicio>.onrender.com/docs`

## Como colocar la regresion real (companeros)
1. Guardar el modelo con `joblib.dump(pipeline, "regresion_popularidad.pkl")`
   (mejor un `Pipeline` que ya incluya el preprocesamiento).
2. Reemplazar `app/modelos/regresion_popularidad.pkl`.
3. Editar `app/modelos/config.json`: `columnas` (nombre y ORDEN exactos del entrenamiento) y `version`.
4. Reemplazar `app/modelos/metricas_regresion.json` con MAE, RMSE y R2.
5. Si usaron otra version de scikit-learn, actualizar `requirements.txt` a esa misma version.
6. `git push` -> Render redespliega solo.

No hay que tocar `main.py` mientras el modelo reciba columnas numericas.

## Columnas (dataset de Spotify)
El servidor viene configurado con estas columnas numericas de audio features de Spotify
(`app/modelos/config.json`). Si la regresion usa otras, solo se edita ese archivo:

`duration_ms, explicit (0/1), danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, time_signature`

La variable a predecir es `popularity` (indice de popularidad). No va en la entrada.
Recomendado: guardar un `Pipeline` de scikit-learn que ya incluya el preprocesamiento.

## Probar la API ya desplegada
```bash
curl -X POST https://TU-SERVICIO.onrender.com/predict/popularidad \
  -H "Content-Type: application/json" \
  -d '{"features":{"duration_ms":210000,"explicit":0,"danceability":0.7,"energy":0.8,"key":5,"loudness":-5,"mode":1,"speechiness":0.05,"acousticness":0.1,"instrumentalness":0,"liveness":0.1,"valence":0.6,"tempo":120,"time_signature":4}}'
```

## Endpoints
| Metodo | Ruta | Para que |
|---|---|---|
| GET | `/health` | estado del servicio y de la BD |
| GET | `/modelo/info` | version, columnas y tipo de modelo |
| POST | `/predict/popularidad` | prediccion (`{"features": {...}}`) |
| GET | `/metricas/regresion` | metricas del modelo |

La semana siguiente: agregar `/predict/es_hit` con su propio PKL y `metricas_clasificacion.json`.
