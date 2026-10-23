# TesisV4

Clasificación binaria de mamografías (BI-RADS 1–2 → 0, BI-RADS 3–5 → 1) con
TensorFlow/Keras. El driver es `multirun.ipynb`: descarga el dataset, arma los
splits y entrena backbones como clasificadores de imagen, clasificadores de
parches y modelos ABMIL (el notebook hoy solo corre `standard` y `resized`).

## Repositorios

| Repo | Rol |
| --- | --- |
| [`emiliodelgadouy/TesisV4`](https://github.com/emiliodelgadouy/TesisV4) | Este workspace: notebook, configuración de la corrida y documentación. |
| [`emiliodelgadouy/TesisV4_src`](https://github.com/emiliodelgadouy/TesisV4_src) | Paquete `src/`. El notebook lo clona en `./src`. Los cambios locales de `src/` no llegan a Colab hasta que se commitean y pushean ahí. |

El proyecto de Comet es `tesis_v4`: todo se reentrena ahí. `tesis_v3` (slug `tesis-v3`) queda como histórico.

## Cómo correr

En Colab, la primera celda de `multirun.ipynb` borra `./src`, clona
`TesisV4_src` y llama a `configure_notebook()` antes de importar
`src.notebook_api`. Fuera de Colab el entorno es `.venv` y las dependencias de
entrenamiento están en `src/requirements.txt` (TensorFlow 2.15/2.16 con CUDA).

El launcher local `tools/run_multirun_a100.sh` abre una sesión Colab A100
high-RAM y ejecuta el notebook. `--reuse` reinicia el kernel de una sesión
viva en vez de crear otra VM.

### Si se cae el ambiente

Cada corrida guarda su avance en su propio experimento de Comet (assets
`resume_*`, vía `CometRunStore`): el estado, los pesos de la
última época y el mejor checkpoint de cada etapa. Para retomar alcanza con
volver a ejecutar el notebook entero, en la misma VM o en otra. Lo terminado
se saltea, la corrida cortada sigue desde la última época guardada y sigue
logueando en el mismo experimento.

- La subida va en segundo plano y no frena el entrenamiento. Si la VM muere
  durante una subida, se retoma desde el último estado completo (se puede
  repetir alguna época).
- Al terminar la corrida se borran los pesos de resume; los finales quedan en
  el modelo `{nombre}_final_weights`.
- Al retomar se recuperan pesos, learning rate y contadores de EarlyStopping y
  ReduceLROnPlateau, pero no los momentos de Adam.
- Una corrida que Comet ya tiene terminada con el mismo nombre se saltea.
- En `src`, `RunStore` guarda lo mismo en una carpeta (por ejemplo, Drive
  montado con `--drive`); `multirun.ipynb` usa siempre Comet.

## Modos

Canónicos, en `src/training/mode.py`:

- `standard`: mamografía completa al tamaño nativo del backbone.
- `resized`: mamografía completa en un canvas de `CONFIG["RESIZED"]["INPUT_SIZES"]`.
- `patch` y `patch_hardneg`: crops `S×S`. Las métricas de val/test usan la ROI; no son rendimiento de mamografía completa.
- `abmil`: bag de grilla `ABMIL.BAG_GRID` (default 3×3) sobre un canvas propio, independiente de RESIZED.
- `abmil_patch_hardneg`: transfiere backbone y proyección densa de `patch_hardneg` e inicializa una salida de bag nueva.

`simple` y `full` siguen parseándose como alias de `standard` y `resized`.
Los nombres de experimento RESIZED llevan el canvas (`resized_672_swint`).

## Estado actual (2026-10-23)

### Código y notebook

`multirun.ipynb` entrena con la etiqueta A (BI-RADS 1–2 vs 3–5) en el proyecto
`tesis_v4`. Es plano: una celda por modo con un `for` sobre `MODELS` que llama a
`train_experiment(mode, backbone, input_size=, seed=)`, en el orden declarado.
`train_experiment` está definida en el notebook (unas diez líneas): arma el nombre, la
saltea si Comet ya la tiene terminada y si no llama a
`run_training_experiment` con un `CometRunStore` para el resume. Solo corre
modos de mamografía completa; ABMIL y parches salieron del notebook (siguen en
`src`). El notebook no usa `ExperimentPlan` ni `RunSpec`.

- `standard`: los 8 `MODELS` al tamaño nativo.
- `resized`: los 8 `MODELS` × 448, 672, 896, 1152 y 1536 (40 corridas). `vits16`
  en 1152 y 1536 corre con batch 1, así que es lo más lento.

Para correr menos se sacan backbones de `MODELS` o canvases de
`RESIZED.INPUT_SIZES`; `SEEDS` (default `[42]`) repite cada corrida con otras
semillas. Si una corrida falla, la celda se corta ahí y al volver a ejecutarla
se reintenta.

Lo que está en Comet manda: una corrida terminada en Comet con el mismo nombre
se saltea y una cortada se retoma. El resumen del final no usa los bucles: lee
de Comet las métricas que cada corrida `standard_*` / `resized_*` terminada (no
archivada) ya logueó (`CometRunIndex.names` + `finished_metrics`), sin
recalcular nada ni usar GPU.

Para monitorear, cada corrida loguea en Comet `run_status`, `run_status_code`,
`run_progress` (0–1 por época) y sus coordenadas `matrix_backbone` /
`matrix_arch` / `matrix_seed` / `matrix_label`.

El notebook solo resume; no compara corridas ni guarda nada. Muestra una tabla
por split (`REPORTE_VAL`, `REPORTE_TEST`) con una fila por corrida y nivel (por
imagen y por mama, máximo entre CC y MLO): discriminación (ROC-AUC, PR-AUC,
PR-AUC/prevalencia, sensibilidad a especificidad 90/95 %, precisión a recall
80 %), métricas con el umbral de Youden de val (sensibilidad, especificidad,
PPV, NPV, F1) y calibración (Brier, ECE). Son las métricas que cada corrida
loguea al terminar (`{split}_*`, `{split}_breast_*`); también loguea la curva
`calibration_{split}` y su IC 95 % por paciente (`test_*_ci_*`). Intervalos y
resúmenes entre semillas se calculan después desde los `predictions_{split}.csv`.

Para que los experimentos se puedan contrastar en otro lugar:

- Todas las corridas se evalúan sobre las mismas imágenes (val y test
  canónicos completos) y la misma tarea (1–2 vs 3–5). La etiqueta
  de entrenamiento (A, B o C, ver `src/dataset/target.py`) es un factor más.
- Las predicciones de cada corrida quedan como asset `predictions_{split}.csv`
  en su experimento de Comet, así cualquier par se compara sin reentrenar
  (`plan.pairwise` sigue disponible en `src` para eso). Como referencia de
  ruido, la misma configuración varió ~0.02 de PR-AUC entre repeticiones.

Para que sean repetibles, el nombre de cada experimento lleva su semilla
(`resized_672_swint_seed42`), y cada corrida fija sus semillas y loguea en Comet la
etiqueta, la tarea, el commit de `src`, el hash de los splits, el hash del
config, las versiones de Python, TensorFlow, Keras y comet_ml, y la GPU. En
Colab no se instalan los requirements, así que la versión de TF la decide
Colab y queda registrada por corrida. Cada corrida falla si algún paciente o
estudio cae en dos splits (`src/training/provenance.py`).

Configuración de la corrida:

- `MODELS`: los 8 backbones que recorren los bucles (`swint`, `vits16`, `efficientnetb0`, `efficientnetb3`, `efficientnetv2b0`, `vgg16`, `chexnet`, `customcnn`). Lo usa también `--backbone` del launcher.
- Canvases RESIZED definidos: 448, 672, 896, 1152 y 1536 (2048 comentado). Son cuadrados y divisibles por 32. El bucle de RESIZED corre todos.
- Batch por canvas (`RESIZED.BATCH_SIZES`), igual para todos los backbones en una A100 80GB: 128, 64, 32, 16 y 8, acotado por el batch calibrado del backbone (`vits16`: 32, 8, 2, 1 y 1). Cache en RAM solo en 448–896.
- `USE_CUSTOM_BATCH_SIZE=True`: si no hay batch explícito por canvas, manda el batch del backbone y se escala con el área (al cuadrado para ViT).
- `METRIC_TO_MAXIMIZE=pr_auc`. El umbral del reporte es Youden sobre validación.
- CLAHE corre sobre la imagen ya lateralizada y antes del resize o del canvas.
- Tres etapas: backbone congelado, fine-tune del último 30 %, fine-tune completo. Cada etapa guarda su mejor checkpoint. Los pesos finales se suben a Comet y se loguea la época global del restore.
- Etiqueta `birads`. `mass` sigue disponible como objetivo legado.

`multirun_wip.ipynb` es una ablación aparte, sin modificar `src/`, que queda
como registro: fijó `resized_448_swint` y comparó tres definiciones de `cls`
(BI-RADS 3 como positivo, excluido, o negativo), con prefijo `wip_` en
`tesis_v3`. `multirun.ipynb` usa solo la etiqueta A.

`optuna_swint672.ipynb` lleva `resized_672_swint` al límite en el proyecto de Comet `tesis_v4_hpo`, también sin modificar `src/`. Los hiperparámetros que `src` tiene fijos (LR por etapa, cabeza, optimizador, label smoothing, aumento, CLAHE, MixUp) se inyectan con hooks en runtime desde el notebook. El notebook tiene tres partes:

- **Búsqueda con Optuna** (~12 h, ~20 trials con poda, objetivo PR-AUC del val canónico). El estudio es un SQLite que se sincroniza como asset `optuna.db` del experimento `optuna_estudio`.
- **Confirmación**: el mejor config con las etiquetas A/B/C × semillas 42/43/44, evaluado siempre sobre la tarea A y comparado contra el baseline de `tesis_v4`.
- **Post-hoc y reporte**: TTA, ensembles y calibración, más observabilidad (Grad-CAM, IG, oclusión y localización contra la ROI). Las tablas van como assets de `estado_hpo`.

### Resultados históricos (`tesis_v3`)

Son de antes de `tesis_v4` y no se mezclan con `tesis_v4`; sirven de
referencia para las estimaciones. `comet_metrics/` (local, fuera de git) tiene el dump del 2026-10-19: 66
experimentos de `tesis-v3`, 38 activos y 28 archivados del barrido anterior
(`simple`/`full`/`abmil`, checkpoint `auc` o `pr_auc` del 07–08/09). Se baja con
`tools/download_comet_metrics.sh --archived` usando el Python del `.venv`.

Mamografía completa, código actual (09/09). Validación es la PR-AUC del
checkpoint; test es la prevalencia del 6.4 % (246/3862):

| Experimento | Val PR-AUC | Test ROC-AUC | Test PR-AUC |
| --- | ---: | ---: | ---: |
| `resized_672_swint` | 0.396 | 0.791 | 0.396 |
| `resized_896_swint` | 0.380 | 0.793 | 0.406 |
| `resized_1152_swint` | 0.351 | 0.771 | 0.373 |
| `resized_448_swint` | 0.339 | 0.772 | 0.361 |
| `standard_swint` | 0.306 | 0.731 | 0.320 |
| `standard_vits16` | 0.290 | 0.750 | 0.288 |
| `standard_efficientnetb3` | 0.257 | 0.693 | 0.287 |

- Swin mejora con el canvas hasta 672–896 y después cae. `resized_1536_swint` quedó cortada en la etapa 2 (val 0.31).
- ViT empeora con canvases grandes (448: 0.742 test ROC; 896: 0.635).
- No hay `resized_*_efficientnet` con el código actual; `multirun.ipynb` las entrena.
- `customcnn` y `chexnet` quedan cerca del azar (test ROC 0.54–0.60).
- La ablación de BI-RADS 3 (`wip_a/b/c`) evaluó cada variante contra un test distinto, así que sus AUC no se comparan entre sí; las etiquetas B y C quedan fuera de `multirun.ipynb`.
- En el barrido archivado, ABMIL queda por debajo de Swin resized (`abmil_patch_hardneg_swint`: 0.767 / 0.358 de test).

### Qué queda por correr

- Las 48 corridas `standard` y `resized` en `tesis_v4` (varias sesiones de Colab con resume).

## Layout

```
multirun.ipynb          driver: standard y resized por backbone, con resume y resumen desde Comet
multirun_wip.ipynb      ablación de etiquetas BI-RADS
optuna_swint672.ipynb   Optuna + observabilidad sobre resized_672_swint (tesis_v4_hpo)
src/                    paquete de entrenamiento (repo aparte; ver src/README.md)
papers/                 papers de referencia
comet_metrics/          dump local de Comet (gitignored)
tools/                  descarga de métricas y launcher Colab (gitignored)
```

`AGENTS.md` describe la arquitectura para quien edita el código.
