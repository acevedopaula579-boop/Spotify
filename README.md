# API de popularidad de canciones

Servidor FastAPI que sirve un modelo de regresion (PKL) y, si hay `DATABASE_URL`,
guarda cada prediccion en PostgreSQL.

## Probar local (opcional)
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://localhost:8000/docs
```

El modelo se carga **una sola vez al arrancar** el proceso, no en cada request.

## Desplegar en Render
1. Subir esta carpeta a un repo de GitHub.
2. Render -> New -> Blueprint -> elegir el repo (lee `render.yaml`: crea API + Postgres).
3. Cuando termine: `https://<tu-servicio>.onrender.com/docs`

## El modelo real (carpeta `regresion/`)
`regresion/models/modelo.pkl` es el artefacto que entreno el equipo de regresion. El
servidor lo sirve desde `app/modelos/regresion_popularidad.pkl` (copia del mismo archivo).

El `.pkl` es un dict con tres claves:

| Clave | Contenido |
|---|---|
| `pipeline` | `Pipeline` de scikit-learn (preprocesamiento + `Ridge`) que se ejecuta en `predict` |
| `ficha` | que predice, con que datos, que tan bien y que decision sostiene |
| `columnas_entrada` | **fuente de verdad** de las columnas y su orden |

`app/main.py` lee `columnas_entrada` del propio `.pkl` al arrancar y valida el esquema
contra ella; `app/modelos/config.json` solo guarda `archivo_modelo` y `version`.
Si cambia el `.pkl`, no hay que tocar el codigo: solo copiar el archivo.

El pipeline referencia `src.variables.derivadas`, asi que el servidor agrega `regresion/`
a `sys.path` antes de cargar el `.pkl`.

Versiones fijadas (las mismas con las que se entreno, ver `regresion/requirements.txt`):
`scikit-learn==1.6.1`, `joblib==1.6.0`, `numpy==2.2.6`, `pandas==2.2.3`, `scipy==1.17.1`,
Python `3.13.16` (fijado en `render.yaml`).

## Entrada: 10 columnas
Todas numericas salvo `genero`, que se valida contra las categorias que conoce el
OneHotEncoder del pipeline (`pop`, `urbano`, `rock`, `electronica`, `indie`); un genero
desconocido devuelve **422**.

`bailabilidad, energia, valencia, acustica` (0 a 1) &middot; `tempo` (pulsos/min) &middot;
`duracion_min` (minutos) &middot; `volumen_db` (decibelios) &middot; `anio` &middot;
`colaboracion` (0/1) &middot; `genero` (texto)

Los rangos numericos se calculan al arrancar desde `regresion/data/canciones.csv` con un
margen del 10% del rango observado, para no rechazar valores reales. Campos desconocidos
tambien dan **422** (`extra="forbid"`).

La variable a predecir es `popularidad` (indice 0-100). No va en la entrada.

## Probar la API
```bash
curl -X POST https://TU-SERVICIO.onrender.com/predict/popularidad \
  -H "Content-Type: application/json" \
  -d '{"bailabilidad":0.72,"energia":0.65,"valencia":0.55,"acustica":0.12,"tempo":118.0,"duracion_min":3.4,"volumen_db":-6.5,"anio":2026,"colaboracion":1,"genero":"urbano"}'
```

## Endpoints
| Metodo | Ruta | Para que |
|---|---|---|
| GET | `/health` | estado del servicio y de la BD |
| GET | `/modelo/info` | nombre, version, columnas y generos del modelo |
| POST | `/predict/popularidad` | prediccion (body plano con los 10 campos) |
| GET | `/metricas/regresion` | metricas del modelo |
| GET | `/predicciones/ultimas` | ultimas predicciones guardadas (requiere BD) |

## Actualizar el modelo
1. Reemplazar `regresion/models/modelo.pkl` (y `regresion/data/canciones.csv` si cambio).
2. Copiarlo a `app/modelos/regresion_popularidad.pkl`.
3. Bajar `version` en `app/modelos/config.json`.
4. Actualizar `app/modelos/metricas_regresion.json` con MAE, RMSE y R2 de la nueva ficha.
5. Si se uso otra version de scikit-learn, actualizar `requirements.txt` **y** la de
   `regresion/requirements.txt` a la misma.
6. `git push` -> Render redespliega solo.

La semana siguiente: agregar `/predict/es_hit` con su propio PKL y `metricas_clasificacion.json`.
