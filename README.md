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

El proyecto de Comet es `tesis_v4`: arranca vacío con la matriz anclada y todo
se reentrena ahí. `tesis_v3` (slug `tesis-v3`) queda como histórico.

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
- Una corrida que Comet ya tiene terminada con el mismo nombre también se
  saltea (`RESUME.SKIP_IF_IN_COMET`).
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

## Estado actual (2026-10-21)

### Código y notebook

`multirun.ipynb` entrena una **matriz anclada**: una celda ancla y un corte por
pregunta de tesis que cambia un solo factor. Las celdas que comparten dos cortes
se entrenan una vez (`plan.unique`). Todo va al proyecto nuevo `tesis_v4`, con
una semilla (42).

- Ancla: `resized_672_swint`, etiqueta A (1–2 vs 3–5).
- Ancla CNN: `resized_672_efficientnetb0`, barata, para ver si un efecto se
  repite fuera de Swin.

| Pregunta | Celdas (etiqueta A salvo en etiqueta) | Corridas nuevas | ~h A100 |
| --- | --- | --- | --- |
| Familia | `resized_672` × {swint, vits16, efficientnetb0, efficientnetv2b0, vgg16, chexnet, customcnn} | 7 | 7 |
| Escala | `resized_672_efficientnetb3` vs `efficientnetb0` | 1 | 1 |
| Resolución | {standard, 448, 672, 896, 1152} × {swint, efficientnetb0} | 8 | 10.5 |
| MIL | `abmil` vs `resized_672`, × {swint, efficientnetb0} | 2 | 2 |
| Preentrenamiento con parches | `abmil_patch_hardneg` (encoder `patch_hardneg`) vs `abmil`, × {swint, efficientnetb0} | 4 | 2.4 |
| Etiqueta | `resized_672` × {swint, efficientnetb0} × {A, B, C} | 4 | 4 |
| Total | | 26 | ~27 nominal, ~35 con margen |

Las horas salen del costo estimado (píxeles relativos a `standard` a 224, ~0.11 h
de A100 por unidad, calibrado con Swin). VGG16 y CheXNet pueden tardar el doble
en 672. `plan.by_cost` corre primero lo barato y `plan.timing` muestra las horas
reales y reestima lo que falta antes de que lleguen 896 y 1152. Quedan fuera a
propósito: 1536, ViT en resolución alta, EfficientNet B1/B2/B4+, V2 S/M/L y
VGG19. El margen del presupuesto de ~48 h es para fallos, inferencia y, más
adelante, 3 semillas en las anclas (`SEMILLAS`).

El reporte va por pregunta, en validación y en test: métricas clínicas de cada
celda y diferencias por pares con IC 95 %, sin elegir ganadores. En familia y
escala se comparan todas contra todas; en resolución, MIL, preentrenamiento y
etiqueta los pares van dentro de cada backbone. El corte de etiqueta también se
evalúa en 1–2 vs 4–5 y muestra la probabilidad mediana por BI-RADS.

Para que los experimentos sean contrastables:

- Todas las corridas se evalúan sobre las mismas imágenes (val y test
  canónicos completos) y la misma tarea (`PLAN.TASK`, 1–2 vs 3–5). La etiqueta
  de entrenamiento (A, B o C, ver `src/dataset/target.py`) es un factor más.
- Las predicciones de cada corrida quedan como asset `predictions_{split}.csv`
  en su experimento de Comet, así cualquier par se compara sin reentrenar.
- Para cada par de corridas se reporta la diferencia de PR-AUC y ROC-AUC con
  IC 95 % por bootstrap pareado agrupado por paciente. Como referencia de
  ruido, la misma configuración varió ~0.02 de PR-AUC entre repeticiones.

Para que sean repetibles, cada corrida fija sus semillas y loguea en Comet la
etiqueta, la tarea, el commit de `src`, el hash de los splits, el hash del
config, las versiones de Python, TensorFlow, Keras y comet_ml, y la GPU. En
Colab no se instalan los requirements, así que la versión de TF la decide
Colab y queda registrada por corrida. Al crear el plan se verifica que ningún
paciente caiga en dos splits.

Los bucles de barrido completo por modo siguen en el notebook, comentados, y
pasan por el mismo plan.

Configuración de la corrida:

- `MODELS`: los 8 backbones de los cortes (`swint`, `vits16`, `efficientnetb0`, `efficientnetb3`, `efficientnetv2b0`, `vgg16`, `chexnet`, `customcnn`). Lo usan también los bucles de barrido y `--backbone` del launcher.
- Canvases RESIZED definidos: 448, 672, 896, 1152 y 1536 (2048 comentado). Son cuadrados y divisibles por 32. La matriz usa hasta 1152.
- Batch por canvas (`RESIZED.BATCH_SIZES`), igual para todos los backbones en una A100 80GB: 128, 64, 32, 16 y 8. Cache en RAM solo en 448–896.
- `USE_CUSTOM_BATCH_SIZE=True`: si no hay batch explícito por canvas, manda el batch del backbone y se escala con el área (al cuadrado para ViT).
- `METRIC_TO_MAXIMIZE=pr_auc`. El umbral del reporte es Youden sobre validación.
- CLAHE corre sobre la imagen ya lateralizada y antes del resize o del canvas.
- Tres etapas: backbone congelado, fine-tune del último 30 %, fine-tune completo. Cada etapa guarda su mejor checkpoint. Los pesos finales se suben a Comet y se loguea la época global del restore.
- Etiqueta `birads`. `mass` sigue disponible como objetivo legado.

`multirun_wip.ipynb` es una ablación aparte, sin modificar `src/`, que queda
como registro: fijó `resized_448_swint` y comparó tres definiciones de `cls`
(BI-RADS 3 como positivo, excluido, o negativo), con prefijo `wip_` en
`tesis_v3`. El corte de etiqueta de la matriz la reemplaza.

### Resultados históricos (`tesis_v3`)

Son de antes de la matriz anclada y no se mezclan con `tesis_v4`; sirven de
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
- No hay `resized_*_efficientnet` con el código actual; la matriz anclada las entrena (familia, escala y resolución).
- `customcnn` y `chexnet` quedan cerca del azar (test ROC 0.54–0.60).
- La ablación de BI-RADS 3 (`wip_a/b/c`) evaluó cada variante contra un test distinto, así que sus AUC no se comparan entre sí; el corte de etiqueta de la matriz las vuelve a entrenar en 672 y las evalúa sobre las mismas imágenes.
- En el barrido archivado, ABMIL queda por debajo de Swin resized (`abmil_patch_hardneg_swint`: 0.767 / 0.358 de test).

### Qué queda por correr

- Las 26 corridas de la matriz anclada en `tesis_v4` (~27 h nominales de A100, varias sesiones de Colab con resume).
- Después, si el presupuesto alcanza, más semillas en las anclas.

## Layout

```
multirun.ipynb          driver: matriz anclada con resume y reporte por pregunta
multirun_wip.ipynb      ablación de etiquetas BI-RADS
src/                    paquete de entrenamiento (repo aparte; ver src/README.md)
papers/                 papers de referencia
comet_metrics/          dump local de Comet (gitignored)
tools/                  descarga de métricas y launcher Colab (gitignored)
```

`AGENTS.md` describe la arquitectura para quien edita el código.
