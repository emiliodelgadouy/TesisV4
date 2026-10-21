# TesisV4

Clasificación binaria de mamografías (BI-RADS 1–2 → 0, BI-RADS 3–5 → 1) con
TensorFlow/Keras. El driver es `multirun.ipynb`: descarga el dataset, arma los
splits y entrena backbones como clasificadores de imagen, clasificadores de
parches y modelos ABMIL.

## Repositorios

| Repo | Rol |
| --- | --- |
| [`emiliodelgadouy/TesisV4`](https://github.com/emiliodelgadouy/TesisV4) | Este workspace: notebook, configuración de la corrida y documentación. |
| [`emiliodelgadouy/TesisV4_src`](https://github.com/emiliodelgadouy/TesisV4_src) | Paquete `src/`. El notebook lo clona en `./src`. Los cambios locales de `src/` no llegan a Colab hasta que se commitean y pushean ahí. |

El proyecto de Comet sigue llamándose `tesis_v3` (slug `tesis-v3`).

## Cómo correr

En Colab, la primera celda de `multirun.ipynb` borra `./src`, clona
`TesisV4_src` y llama a `configure_notebook()` antes de importar
`src.notebook_api`. Fuera de Colab el entorno es `.venv` y las dependencias de
entrenamiento están en `src/requirements.txt` (TensorFlow 2.15/2.16 con CUDA).

El launcher local `tools/run_multirun_a100.sh` abre una sesión Colab A100
high-RAM y ejecuta el notebook. `--reuse` reinicia el kernel de una sesión
viva en vez de crear otra VM.

### Si se cae el ambiente

Cada corrida guarda su avance en su propio experimento de Comet
(`RESUME.BACKEND="comet"`, assets `resume_*`): el estado, los pesos de la
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
- Una corrida que Comet ya tiene terminada desde `RESUME.COMET_DONE_SINCE`
  también se saltea (`RESUME.SKIP_IF_IN_COMET`).
- `RESUME.BACKEND="dir"` guarda lo mismo en la carpeta `RESUME.DIR` (por
  ejemplo, Drive montado con `--drive`).

## Modos

Canónicos, en `src/training/mode.py`:

- `standard`: mamografía completa al tamaño nativo del backbone.
- `resized`: mamografía completa en un canvas de `CONFIG["RESIZED"]["INPUT_SIZES"]`.
- `patch` y `patch_hardneg`: crops `S×S`. Las métricas de val/test usan la ROI; no son rendimiento de mamografía completa.
- `abmil`: bag de grilla `ABMIL.BAG_GRID` (default 3×3) sobre un canvas propio, independiente de RESIZED.
- `abmil_patch_hardneg`: transfiere backbone y proyección densa de `patch_hardneg` e inicializa una salida de bag nueva.

`simple` y `full` siguen parseándose como alias de `standard` y `resized`.
Los nombres de experimento RESIZED llevan el canvas (`resized_672_swint`).

## Estado actual (2026-10-19)

### Código y notebook

`multirun.ipynb` corre un plan por fases en vez del barrido completo. Se fija
una referencia y se mueve un factor por vez. Para que los experimentos sean
contrastables:

- Todas las corridas se evalúan sobre las mismas imágenes (val y test
  canónicos completos) y la misma tarea (`PLAN.TASK`, 1–2 vs 3–5). La etiqueta
  de entrenamiento (A, B o C, ver `src/dataset/target.py`) es un factor más.
- Las predicciones de cada corrida quedan como asset `predictions_{split}.csv`
  en su experimento de Comet, así cualquier par se compara sin reentrenar.
- Un candidato gana si mejora la PR-AUC de validación por más de
  `PLAN.MARGIN` (0.02, ruido entre semillas) y el IC 95 % pareado de la
  diferencia, con bootstrap agrupado por paciente, queda por encima de 0.
- Test se mira recién en la fase 4, como media ± desvío entre semillas.

Para que sean repetibles, cada corrida fija sus semillas y loguea en Comet la
etiqueta, la tarea, el commit de `src`, el hash de los splits y el hash del
config. Al crear el plan se verifica que ningún paciente caiga en dos splits.

| Fase | Pregunta | Corridas |
| --- | --- | --- |
| Ref. | Ya en Comet | `resized_672_swint` |
| 1 | ¿Otra etiqueta de BI-RADS 3 mejora la tarea clínica? | Ninguna: usa los pesos `wip_*` (`PLAN.ALIASES`) |
| 2 | ¿Algún EfficientNet le gana a Swin? | EfficientNet B1 y B3 en 672, con la etiqueta ganadora |
| 3 | ¿ABMIL le gana a la imagen entera? | `patch_hardneg` → `abmil_patch_hardneg` con el backbone ganador |
| 4 | Confirmar | Los 2 mejores × 3 semillas, con test |

`multirun_wip.ipynb` queda como registro de cómo se entrenaron las corridas
`wip_*`; el plan ya cubre etiquetas, predicciones exportadas, métricas
clínicas, bootstrap agrupado y chequeo de leakage.

Los bucles de barrido completo por modo siguen en el notebook, comentados, y
pasan por el mismo plan.

Configuración de la corrida:

- `MODELS`: `vits16`, `swint`, `efficientnetb0`, `efficientnetb1`, `efficientnetb2`, `efficientnetb3` (lo usan los bucles de barrido completo y `--backbone` del launcher). `chexnet` y `vgg19` quedan comentados.
- Canvases RESIZED: 448, 672, 896, 1152 y 1536. 2048 está definido y comentado. Son cuadrados y divisibles por 32.
- Batch por canvas (`RESIZED.BATCH_SIZES`), igual para todos esos backbones en una A100 80GB: 128, 64, 32, 16 y 8. Cache en RAM solo en 448–896.
- `USE_CUSTOM_BATCH_SIZE=True`: si no hay batch explícito por canvas, manda el batch del backbone y se escala con el área (al cuadrado para ViT).
- `METRIC_TO_MAXIMIZE=pr_auc`. El umbral del reporte es Youden sobre validación.
- CLAHE corre sobre la imagen ya lateralizada y antes del resize o del canvas.
- Tres etapas: backbone congelado, fine-tune del último 30 %, fine-tune completo. Cada etapa guarda su mejor checkpoint. Los pesos finales se suben a Comet y se loguea la época global del restore.
- Etiqueta `birads`. `mass` sigue disponible como objetivo legado.

`multirun_wip.ipynb` es una ablación aparte, sin modificar `src/`. Fija
`resized_448_swint` y compara tres definiciones de `cls` (BI-RADS 3 como
positivo, excluido, o negativo), con prefijo `wip_` en Comet. Reutiliza
`splits/dataset_splits.json`.

### Resultados en Comet

`comet_metrics/` (local, fuera de git) tiene el dump del 2026-10-19: 66
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
- No hay `resized_*_efficientnet` con el código actual; es la fase 1.
- `customcnn` y `chexnet` quedan cerca del azar (test ROC 0.54–0.60).
- La ablación de BI-RADS 3 (`wip_a/b/c`) evaluó cada variante contra un test distinto, así que sus AUC no se comparan entre sí; la fase 1 las reevalúa sobre las mismas imágenes.
- En el barrido archivado, ABMIL queda por debajo de Swin resized (`abmil_patch_hardneg_swint`: 0.767 / 0.358 de test).

### Qué queda por correr

- Fases 1–4 del plan (~12–15 h de A100).
- El canvas 2048, 1536 completo, ViT en canvases grandes y los backbones pesados (`chexnet`, `vgg19`) quedan fuera del plan.

## Layout

```
multirun.ipynb          driver: plan por fases con resume
multirun_wip.ipynb      ablación de etiquetas BI-RADS
src/                    paquete de entrenamiento (repo aparte; ver src/README.md)
papers/                 papers de referencia
comet_metrics/          dump local de Comet (gitignored)
tools/                  descarga de métricas y launcher Colab (gitignored)
```

`AGENTS.md` describe la arquitectura para quien edita el código.
