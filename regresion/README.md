<div align="center">

# Predicción de popularidad de canciones

Repositorio base del taller en dos sprints: regresión lineal y clasificación binaria

[![Python](https://img.shields.io/badge/Python-3.11%20a%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)](https://render.com)

</div>

---

Una disquera debe decidir, antes del lanzamiento, en qué canciones invertir promoción. Este repositorio contiene un servicio de predicción listo para publicar y el código con el que se entrenaron los modelos de referencia de los dos sprints del taller.

El servicio lee un único archivo, `models/modelo.pkl`, con el modelo entrenado y su ficha. En cada sprint usted reemplaza ese archivo por el de su propio modelo y publica el servicio en Render. El código del servicio no se toca, porque todo lo que necesita para responder está en la ficha.

## Los dos sprints

Cada sprint cierra con un incremento en producción: su servicio publicado en Render con un modelo propio.

| | Sprint 1: regresión lineal | Sprint 2: clasificación binaria |
|---|---|---|
| Objetivo del sprint | Estimar la popularidad de una canción nueva y recomendar si se promociona | Decidir la promoción según lo que cuesta equivocarse |
| Pregunta | ¿Qué popularidad alcanzará la canción? | ¿La canción será un hit? |
| Objetivo del modelo | `popularidad` (0 a 100) | `es_hit` (1 si la popularidad llega a 70) |
| Variables excluidas | `reproducciones_sem1`, `es_hit` | `reproducciones_sem1`, `popularidad` |
| Modelo de referencia | Regresión lineal con duración² | Regresión logística |
| Resultado de referencia (2020-2025) | MAE 5,75; la línea base obtiene 10,44 | Costo 99; la línea base obtiene 318 |
| Regla de decisión | Promocionar si la popularidad esperada supera 65 | Promocionar si la probabilidad de hit supera 0,27 |
| Código de entrenamiento | `src/entrenar_regresion.py` | `src/entrenar_clasificacion.py` |

El sprint 1 está disponible y el repositorio trae su modelo de referencia en `models/modelo.pkl`. El sprint 2 usa el mismo servicio con otro archivo de modelo. En ambos se entrena con las canciones de 1995 a 2019 y se evalúa con las de 2020 a 2025.

## Cómo trabajar cada sprint

1. Clone el repositorio e instale las dependencias, como se explica en [Ejecución local](#ejecución-local).
2. Entrene su modelo y guárdelo en `models/modelo.pkl` con su ficha.
3. Compruebe con `pytest -q` que el servicio lo carga y responde.
4. Suba el cambio a su repositorio en GitHub y publique el servicio en Render.
5. Revise la definición de terminado y entregue la URL del servicio.

En el sprint 2 se repiten los pasos 2 a 5 con el clasificador. La URL no cambia.

### Definición de terminado

- [ ] `pytest -q` pasa con su `models/modelo.pkl`.
- [ ] La ficha (`GET /modelo`) reporta las métricas de su propia evaluación sobre 2020-2025, y su modelo supera la línea base.
- [ ] El servicio está publicado en Render con su modelo.
- [ ] `python -m src.probar_servicio SU_URL` termina con "servicio en orden". Con esta prueba se califica.

## Crear su modelo

La forma más directa es editar `construir_pipeline()` y `NOMBRE` en el script del sprint y ejecutarlo. El script se encarga del resto y deja el archivo en `models/modelo.pkl`.

```bash
python -m src.entrenar_regresion       # sprint 1
python -m src.entrenar_clasificacion   # sprint 2
```

Si prefiere trabajar en su notebook, importe las funciones del repositorio para que el servicio pueda cargar lo que usted guarde:

```python
import sys
sys.path.append("ruta/a/demo-reg-clas-canciones")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from src.variables import COLUMNAS_ENTRADA, derivadas, preprocesamiento
from src.artefacto import guardar

mi_pipeline = make_pipeline(FunctionTransformer(derivadas), preprocesamiento(), MiEstimador())
mi_pipeline.fit(X_entrenamiento[COLUMNAS_ENTRADA], y_entrenamiento)

guardar(mi_pipeline, mi_ficha, "ruta/a/demo-reg-clas-canciones/models/modelo.pkl")
```

Antes de escribir el archivo, `guardar()` revisa la ficha y prueba el pipeline. Si falta algo, el mensaje de error lo indica.

### Requisitos del archivo

El pipeline recibe las 10 columnas crudas de `COLUMNAS_ENTRADA` (`src/variables.py`) y hace dentro todo el preprocesamiento. Una variable derivada nueva se agrega en `derivadas()` de `src/variables.py`, no en el notebook; de lo contrario el servicio no la encuentra al cargar el archivo.

La ficha debe tener `nombre`, `tarea` (`"regresion"` o `"clasificacion"`), `objetivo`, `momento_prediccion`, `variables_excluidas`, `linea_base`, `desempeno` y `decision`. En regresión, `desempeno` lleva `mae` y `decision` lleva `corte_promocion`. En clasificación, `decision` lleva `umbral` y el modelo debe tener `predict_proba`. Los scripts de `src/` traen una ficha completa de cada tipo.

Entrene con las versiones de `requirements.txt`, porque un `.pkl` creado con otra versión de scikit-learn puede no cargar en Render. Si trabaja en Google Colab, clone el repositorio e instale esas versiones en la primera celda:

```python
!git clone https://github.com/ebuitrago/demo-reg-clas-canciones.git
%pip install -q -r demo-reg-clas-canciones/requirements.txt
```

Luego reinicie la sesión (menú Entorno de ejecución > Reiniciar sesión) para que Colab use las versiones instaladas. En Colab la ruta del repositorio es `/content/demo-reg-clas-canciones`.

El archivo debe pesar menos de 20 MB. Un Random Forest sin límite de profundidad puede pasarse.

## API

| Método | Ruta | Respuesta |
|--------|------|-----------|
| `GET` | `/docs` | Documentación interactiva, donde se pueden probar todas las rutas |
| `GET` | `/salud` | Estado del servicio y modelo cargado |
| `GET` | `/modelo` | Ficha del modelo |
| `POST` | `/predecir` | Predicción y decisión para una canción |
| `POST` | `/predecir_lote` | Lo mismo para hasta 1.000 canciones |

Ejemplo de entrada para `/predecir`:

```json
{"bailabilidad": 0.72, "energia": 0.65, "valencia": 0.55, "acustica": 0.12, "tempo": 118,
 "duracion_min": 3.4, "volumen_db": -6.5, "anio": 2026, "colaboracion": 1, "genero": "urbano"}
```

Respuesta con el modelo del sprint 1:

```json
{"tarea": "regresion", "modelo": "lineal + duración² (2026-09-27)",
 "popularidad_esperada": 84.8, "rango": [79.1, 90.6], "corte_promocion": 65.0,
 "probabilidad_hit": null, "umbral": null,
 "promocionar": true, "explicacion": "popularidad esperada 84.8 (±5.75) supera el corte 65"}
```

Con el modelo del sprint 2, la misma entrada trae valores en `probabilidad_hit` y `umbral`, y `popularidad_esperada`, `rango` y `corte_promocion` llegan en `null`. Si la entrada trae un campo que no existe antes del lanzamiento, como `reproducciones_sem1`, el servicio responde `422`.

## Estructura

```
demo-reg-clas-canciones/
├── data/canciones.csv              # 2.000 canciones sintéticas, 1995-2025
├── models/modelo.pkl               # El archivo que usted reemplaza en cada sprint
├── src/
│   ├── variables.py                # Columnas de entrada, variables derivadas y partición temporal
│   ├── artefacto.py                # guardar() y cargar() del modelo
│   ├── entrenar_regresion.py       # Sprint 1
│   ├── entrenar_clasificacion.py   # Sprint 2
│   ├── probar_servicio.py          # Prueba de humo contra el servicio publicado
│   └── app/
│       ├── esquemas.py             # Formato de entrada y salida
│       └── main.py                 # Servicio FastAPI
├── tests/test_servicio.py          # Pruebas
├── requirements.txt                # Dependencias con versiones fijas
└── render.yaml                     # Configuración para Render (Python 3.11)
```

## Ejecución local

Necesita Python 3.11, 3.12 o 3.13. **No use Python 3.14 ni 3.10 o anteriores**: con esas versiones la instalación falla. Compruebe su versión con `python --version`; en Windows, `py --list` muestra las versiones instaladas. Si solo tiene 3.14, instale también la 3.13 desde <https://www.python.org/downloads/> y cree el entorno con ella.

```bash
git clone https://github.com/ebuitrago/demo-reg-clas-canciones.git
cd demo-reg-clas-canciones
python -m venv .venv                 # Windows, eligiendo versión: py -3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                            # comprueba que models/modelo.pkl carga y responde
uvicorn src.app.main:app --reload    # http://127.0.0.1:8000/docs
```

Con el servicio arriba, en otra terminal:

```bash
python -m src.probar_servicio
```

## Publicar en Render

Render tiene un plan gratuito que no pide tarjeta de crédito.

1. Cree en su cuenta de GitHub un repositorio propio con este contenido y su `models/modelo.pkl`. Puede usar **Fork** o crear un repositorio nuevo y subir los archivos.
2. Entre a <https://render.com> con su cuenta de GitHub.
3. En el panel elija **New +**, luego **Blueprint**, seleccione su repositorio y confirme con **Apply**. Render lee `render.yaml` y tarda entre 3 y 5 minutos en construir el servicio.
4. Copie la URL que asigna Render y compruébela:

   ```bash
   python -m src.probar_servicio https://su-servicio.onrender.com
   ```

Render vuelve a publicar el servicio cada vez que usted sube un cambio a GitHub. Para el sprint 2 basta con subir el nuevo `models/modelo.pkl`.

En el plan gratuito el servicio se suspende tras 15 minutos sin uso, y la primera petición después de eso tarda cerca de un minuto. La prueba de humo espera hasta 90 segundos.

## Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| `pip install` falla con "No matching distribution found" | Su versión de Python no es compatible. Cree el entorno con Python 3.11, 3.12 o 3.13 |
| En Windows, PowerShell no deja ejecutar `activate` | Ejecute una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelva a intentar |
| Aviso de que el modelo se creó con otra versión de scikit-learn | Reentrene con las versiones de `requirements.txt` |
| `503` en `/modelo` o `/predecir` | Falta `models/modelo.pkl` en el repositorio |
| `422` en `/predecir` | La entrada no cumple el formato de `src/app/esquemas.py`; el detalle de la respuesta indica el campo |
| `AttributeError` al cargar el modelo | El pipeline usa una función que no está en `src/variables.py`. Muévala allí y reentrene |
| `ModuleNotFoundError: No module named 'src'` | Ejecute los comandos desde la raíz del repositorio |
| La primera petición a Render tarda o falla | El servicio estaba suspendido. Repita en un minuto |

---

<div align="center">
<sub>Universidad Central, Ingeniería de Sistemas, 2026-2. Prof. Elias Buitrago Bolivar</sub>
</div>
