# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo is

A thesis codebase for breast-cancer classification from mammography images. It trains
TensorFlow/Keras backbones as image classifiers, supervised patch classifiers, and ABMIL
multiple-instance classifiers. Spanish comments and docstrings in `src/` are intentional; match
that convention when editing source code.

`src/` is a standalone Git repository consumed by the notebook as
`emiliodelgadouy/TesisV4_src` (`https://github.com/emiliodelgadouy/TesisV4_src`). The Comet
project name in the notebook is still `tesis_v3` (slug `tesis-v3`). The current workspace
driver is `multirun.ipynb`; its Colab setup recreates `./src` from that remote, so local
`src/` edits are not visible there until they are committed and pushed.

## Current status (2026-10-19)

The human-facing snapshot lives in the workspace `README.md` and `src/README.md`. Facts that
change how to edit or run the code:

- `multirun.ipynb` trains an explicit matrix `MODELS` × architectures × `SEMILLAS` (label A):
 `MODELS` holds one representative per family (`swint`, `vits16`, `efficientnetb0`,
 `efficientnetb3`, `efficientnetv2b0`, `vgg16`, `chexnet`, `customcnn`); architectures are
 `standard`, `resized` 448–1536, `abmil`, `abmil_patch_hardneg` (encoder
 `patch_hardneg_{backbone}`), minus `EXCLUIR` (`vits16` at 1152/1536). Training labels B and
 C are crossed only on the cheap columns `ARQ_ETIQUETA` = {`standard`, `resized_448`} for all
 backbones (`resized_448_swint` B/C are the `wip_*` runs via `PLAN.ALIASES`). Runs go
 cheapest first (`plan.by_cost`) with `continue_on_error=True`; failures are listed, retried
 on the next run. Results are reported raw (`plan.matrix`, `plan.evaluate`, `plan.pairwise`)
 on val and test: labels within each backbone×arch, and backbones/architectures within
 label A (`MATRIZ_A`) by column and row, plus `plan.seed_summary`. No automatic winners,
 finalists or propagated decisions — the user asked for raw comparisons. Every
 comparison uses the same canonical images and the fixed task `PLAN.TASK` ("A"), with
 predictions cached as Comet assets `predictions_{split}.csv` (logged by
 `TrainingExperiment` at the end of each run; recomputed from final weights for older runs
 or when the run's rows differ from the canonical split). The
 per-mode full-sweep loops are commented and also go through `plan.run`.
- Runs log `LABEL*`, `EVAL_TASK` and `PROVENANCE_*` (src commit, split hash, config hash).
 `multirun_wip.ipynb` is kept as the record of how the `wip_*` runs were trained.
- Resume: `RESUME.BACKEND="comet"` stores each run's state and weights as `resume_*` assets of
 its own Comet experiment (`"dir"` uses a folder in `RESUME.DIR`). Re-running the notebook
 skips `done` runs, resumes a cut run from its last epoch, and keeps logging into the same
 Comet experiment. `SKIP_IF_IN_COMET` + `COMET_DONE_SINCE` (2026-09-09) also treat finished
 Comet runs as done.
- `MODELS`: `vits16`, `swint`, `efficientnetb0`–`efficientnetb3` (used by the commented
 sweep loops and the launcher `--backbone`). RESIZED canvases in the notebook: 448, 672,
 896, 1152, 1536 (2048 is listed and commented), with `RESIZED.BATCH_SIZES` and
 `RESIZED.CACHE` (cache on for 448–896).
- Checkpoint metric is `GENERAL.METRIC_TO_MAXIMIZE = "pr_auc"`. CLAHE already runs after
  lateralization and before resize/canvas. Final weights are uploaded to Comet
  (`log_model`) and the global restore epoch is logged.
- `multirun_wip.ipynb` is a side ablation of the BI-RADS 3 label on `resized_448_swint`. It
  must not modify `src/`. Experiments use a `wip_` name prefix.
- The local `comet_metrics/` dump (2026-10-19, 66 runs including 28 archived) mixes the
 current names with the archived `simple` / `full` / `abmil*` sweep from 2026-09-07/08.
 Only non-archived runs from 2026-09-09 on match the current code. `comet_metrics/` and
 `tools/` are gitignored in the workspace repo. `tools/run_multirun_a100.sh` has `--drive`
 (mounts Drive via `colab drivemount`, interactive) and `--reuse` (restart kernel, no new VM).

## Git

Work on `main` in both repos (`TesisV4` and `src/` → `TesisV4_src`). Do not create feature
branches; commit to `main` and push when the user asks. Colab clones `TesisV4_src` `main`, so
`src/` changes must be pushed there to take effect.

## Environment and validation

- The Python environment is normally `.venv`. Train-time dependencies live in
  `src/requirements.txt`: TensorFlow 2.15/2.16 with CUDA support, headless
  OpenCV (CLAHE), pandas, scikit-learn, Comet ML, gdown, requests, pillow, matplotlib, and
  psutil. GCS downloads prefer anonymous `gsutil` when crcmod is compiled, then parallel
  HTTPS `curl`, then `gcloud storage`, single-stream curl, and `requests`.
- `pyproject.toml` only configures Black/Ruff with `line-length = 300`. There is no CI or test
  suite, so validate changes with focused scripts/imports and the relevant notebook cells.
- Call `src.notebook.configure_notebook()` before importing `src.notebook_api`. Outside Colab it
  installs requirements first, then imports Comet before TensorFlow, applies the mixed-precision
  policy, and enables IPython autoreload. Importing/installing dependencies after TensorFlow is
  already loaded can leave the running process on stale package versions.
- Do not commit Python bytecode or cache directories. `src/.gitignore` excludes `*.py[cod]` and
  `__pycache__/`.

## Dataset (`src/dataset/`)

- `config.py` / `DatasetConfig` owns local paths and GCS URIs.
- `repository.py` / `DatasetRepository` downloads `square_images.tar.gz`, `square_data.csv`, and
  `dataset_splits.json` from `gs://helen-data`, builds image paths, filters to BI-RADS 1-5,
  and derives binary `cls` (1-2 → 0, 3-5 → 1). `download_and_build(config)` is the I/O entry
  point; its `reduced=True` / `GENERAL.REDUCED_DATASET` option is only for quick pipeline
  checks, not a production split.
- `target.py` / `TargetMode.from_config(config)` + `masks(df)` resolves `birads` vs `mass`
  (`full` is a legacy alias of `birads`).
- `preparer.py` / `DatasetPreparer.prepare(config)` downloads, labels, and deduplicates.
- `splits.py` / `SplitManager.get(config, ds)` loads or regenerates train/val/test (and
  undersamples train when regenerating).
- `provider.py` / `DatasetProvider` builds `tf.data` pipelines. Patch crops support `uniform`,
  `normal`, `roi`, and `avoid_roi`, globally or per label. MIL bags are controlled by mode,
  `ABMIL.BAG_GRID`, `ABMIL.BAG_CANVAS_MODE`, and `ABMIL.BAG_KERAS_TILING`; dataset and model settings must agree.
  `InspectDataset` / `DatasetSplits` wrap `tf.data` so evaluation can use a stable order.
  `as_tf_dataset()` unwraps an `InspectDataset` before passing it to Keras.
- `images.py` / `ImageDecoder` decodes JPEG/PNG. CLAHE (same `clip_limit`/`tile_grid`) runs after lateralization and before resize/canvas in `_process` / `_bag_canvas`.
- `mixup.py` / `PositiveMixup` applies MixUp only among positive samples.

ROI coordinates used by patch mode come from the precomputed
`pad_resized_{xmin,ymin,xmax,ymax}_norm` CSV columns. The current CSV contains a small number
of out-of-bounds or inverted transformed boxes. They already exist in the upstream crop/resize
metadata (some appear to cross the crop boundary); the provider does not generate those values.

## Training modes (`src/training/mode.py`)

Canonical modes are:

- `standard`
- `resized`
- `patch`
- `patch_hardneg`
- `abmil`
- `abmil_patch_hardneg`

`TrainingMode.parse(name)` accepts case, hyphen, and underscore variants and returns the
canonical string (`"abmil"`) for Comet and CONFIG. `TrainingExperiment.run()` parses once on
entry; factory and provider code also parse their inputs. `simple` and `full` remain parse
aliases of `standard` and `resized`.

`standard` uses the backbone native input size. `resized` picks a canvas from the discrete
list `CONFIG["RESIZED"]["INPUT_SIZES"]` (integers or `(H, W)` pairs). A run can pass
`input_size=` or a variant name such as `resized_672` / `resized_672x672`; both must belong
to that list. If the list is omitted, the default is 3× the backbone native size. Experiment
names include the canvas (`resized_672_swint`) so distinct sizes do not collide.
ABMIL reads `CONFIG["ABMIL"]` (`BAG_GRID`, `BAG_CANVAS_MODE`, attention, batch, tiling);
`CONFIG["MIL"]` remains a legacy alias. ABMIL tiles a canvas of `BAG_GRID * backbone_native`,
independently of RESIZED. Tiling is in-dataset or in-Keras according to `BAG_KERAS_TILING`.

### PATCH semantics

The default patch policy is label `1` -> crop around ROI and label `0` -> crop outside ROI.
This is supervised construction of the patch task, not leakage between patients or splits.
Consequently, PATCH validation/test metrics are ROI-assisted patch-level metrics and must not be
reported as end-to-end whole-mammogram performance. The ABMIL modes are the whole-image/bag
evaluation path; patch pretraining can initialize their encoder.

`patch_hardneg` adds negative patches sampled outside the ROI from positive mammograms and can
align crops to the later ABMIL grid.

`abmil_patch_hardneg` transfers the patch backbone and dense instance projection, but always
initializes a new bag classifier. It does not reuse the instance classifier as the global bag
output.

## Architecture

### Notebook facade

- `src/notebook.py` / `NotebookRuntime` + `configure_notebook()` boots the runtime.
- `src/notebook_api.py` is the public notebook facade. When adding a notebook-facing helper,
  update both its import and `__all__`.
- `src/training/experiment.py` / `TrainingExperiment.run()` is the end-to-end driver used by
  `multirun.ipynb`. It runs three training stages, evaluates splits, selects thresholds, logs
  results, frees resources, and re-raises failures after cleanup so a failed run cannot appear
  successful.

### Backbones (`src/backbones/`)

- Each module defines a `Backbone` subclass auto-registered by `Backbone.__init_subclass__` in
  `base.py`. `Backbone` is an ABC: a concrete provider must implement `preprocess_input` and
  `build`. Keras-application families (`efficientnet.py`, `efficientnet_v2.py`, `vgg.py`)
  inherit those from `ImagenetBackbone`; custom extractors (`vit.py`, `swin.py`, `chexnet.py`,
  `custom_cnn.py`, `custom_tiny.py`) override them in their own file.
- `resolve_backbone(name, input_size=None)` returns `(keras_model, preprocess_fn, input_size)`.
- Each provider declares `batch_size` per training mode (`standard`, `resized`, `patch`,
  `patch_hardneg`, `abmil`, `abmil_patch_hardneg`). Those values are the largest
  power-of-two batch that fits an A100 80GB (mixed_float16, stage 3) at the
  reference geometry: native `S` for STANDARD/PATCH, `RESIZED` 672, ABMIL 3×3.
  `resolve_batch_size()` then scales with the chosen RESIZED canvas (area; squared for
  ViT attention) and `ABMIL.BAG_GRID` so a larger canvas/grid does not OOM.
  If `CONFIG["RESIZED"]["BATCH_SIZES"]` maps a canvas to a batch, that value is used
  as-is for every backbone (a single safe occupancy, not per-backbone).
  `CONFIG["RESIZED"]["CACHE"]` does the same for in-memory `tf.data` cache (on for
  448–896, off for 1152+ so host RAM does not OOM). `GENERAL.CACHE_DATASET` is the
  master switch.
  `GENERAL.USE_CUSTOM_BATCH_SIZE` switches between those values and the global
  `GENERAL.BATCH_SIZE` / `ABMIL.BATCH_SIZE`.
- Medical-weight caches live in `medical_weights.py` (`medical_weights_dir()`); CheXNet downloads
  its weights there with `gdown`.
- `vit.py` / `ViTS16Backbone` (`vits16`) is ViT-S/16 with official AugReg ImageNet-21k→1k
  weights. It reshapes patch tokens to a spatial map so GAP and ABMIL stay unchanged; RESIZED
  interpolates positional embeddings when the canvas is not 224. Global attention is quadratic
  in the number of patches, so it is a poor fit for large RESIZED canvases.
- `swin.py` / `SwinTBackbone` (`swint`) is Swin-T with ImageNet-1k weights. Windowed attention
  scales linearly with resolution; the last stage is a spatial map `H/32 × W/32 × 768` (7×7 at
  224, 21×21 at 672). Prefer `swint` over `vits16` for RESIZED / high-resolution inputs.

### Model builders (`src/model_builder/`)

- `factory.py` / `ModelBuilderFactory.create()` / `ModelBuilder` dispatches by parsed mode.
  `StandardModelBuilder` and `ResizedModelBuilder` share the image graph but override
  input-size policy: native backbone size vs a canvas from
  `CONFIG["RESIZED"]["INPUT_SIZES"]` (plus experiment name, cache default, and Keras
  model name). Patch modes reuse `ImageClassifierBuilder`; ABMIL modes reuse
  `AbmilModelBuilder`. `abmil_patch_hardneg` requires a patch encoder; transfer
  happens in `after_build`.
- `base.py::BaseModelBuilder` is an ABC: it owns the Keras graph template
  (`inputs` → `encode_features` → `aggregate` → `regularize` → `output`),
  augmentation, loss/metrics, and freeze/unfreeze. Subclasses must implement
  `inputs`, `encode_features`, and `aggregate`.
- `src/training/model_trainer.py` / `ModelTrainer` owns compile, fit, callbacks,
  checkpoints, evaluate, and predict. The builder delegates those calls.
- `mil_base.py` / `MilModelBuilderBase` overrides the graph hooks and requires
  `pool_instances`. Transfer patch→bag happens in `after_build`.
- `abmil.py` implements gated attention pooling; `image.py` is the shared image graph;
  `standard.py` and `resized.py` override native vs canvas size policy.
- `layers.py` contains `BagTiling` and `GatedAttentionPooling`.

Checkpoints are written under `checkpoints/`. Experiment runs pass a
`{mode}_{backbone}` prefix (`resized_{H}_{backbone}` when the canvas is square), producing
`{prefix}_stage_{n}_epoch{NN}.weights.h5`; without a prefix the stem is
`stage_{n}_epoch{NN}.weights.h5`. Each stage retains its best checkpoint.

### Training (`src/training/`)

- `mode.py` / `TrainingMode`
- `experiment.py` / `TrainingExperiment`
- `model_trainer.py` / `ModelTrainer` — compile/fit/checkpoints for a built Keras graph
- `stage_runner.py` / `TrainingStageRunner` — setup, warmup, fit, best checkpoint for that
  stage; it does not evaluate the model
- `timer.py` / `TrainingTimer`, `EpochTimer`
- `evaluator.py` / `Predictor`, `ThresholdSelector`, `PairedBootstrap`
- `resources.py` / `GpuResources`, `RandomSeeds`
- `resume.py` / `RunStore` (folder), `TrainingResume`, `ResumeCallback`. A store's `commit`
 uploads the weights the state references, then the state, then deletes unreferenced
 weights; `src/tracking/comet_store.py` / `CometRunStore` does the same with versioned
 `resume_state_{N}.json` assets via the REST API in a background thread that keeps only the
 newest pending commit. `ResumeCallback` must stay last in the callback list (after the
 checkpoint, EarlyStopping and ReduceLROnPlateau callbacks that `ModelTrainer.callbacks()`
 exposes as `early_stopping` / `reduce_lr`). Resume restores weights, LR and callback
 counters, not Adam moments. Stage epochs are counted from `History.epoch` (fits may start
 at `initial_epoch > 0`), not `len(history["loss"])`.
- `plan.py` / `ExperimentPlan`, `RunSpec`, `prepare_resume` — runs specs, skips finished
 work (store or Comet), relabels frames per `RunSpec.label` (`dataset/target.py`
 `LABEL_SCHEMES` / `apply_label_scheme`), and compares on canonical-split predictions
 (`ClinicalMetrics`, `PairedBootstrap` in `evaluator.py`). `resolve_experiment_name()` in
 `experiment.py` is the single source for run names; suffixes are `_lbl{x}` (non-A label)
 then `_seed{N}`. `RandomSeeds.set` runs per spec so results do not depend on run order.
- `inference.py` / `build_trained_model`, `predict_with_weights` — rebuild a mode×backbone
 graph from a `.weights.h5` (also used to transfer an already-trained `patch_hardneg`).

### Tracking (`src/tracking/`)

- `comet.py` / `CometTracker` groups login, start, samples, eval, and the final report.
  `CometEpochLogger` is the Keras callback. Comet must be imported before TensorFlow for Keras
  instrumentation. Input-sample logging keeps a bounded, balanced candidate set; it must not
  concatenate an entire dataset just to upload a few images. The run logs
  `METRIC_TO_MAXIMIZE`, the global restore epoch (`restore_epoch` / `val_best_metric`), and
  the final `.weights.h5` via `log_model`. `CometTracker.start(experiment_key=...)` reopens an
  existing experiment (`mode="get"`) on resume and skips re-uploading input samples.
- `comet_results.py` / `CometRunIndex` — finished non-archived runs by name (has
  `test_roc_auc` and `val_best_metric`), optional `done_since` date cut, and download of the
  newest `{name}_final_weights` asset (a resumed run may upload it twice).

## Working with the notebook

`multirun.ipynb` is the current experiment driver. Its flow is: clone/setup `src` from
`TesisV4_src` -> `configure_notebook()` -> import `src.notebook_api` -> define config
(including `RESUME` and `PLAN`) -> `prepare_resume()` -> download/prepare data ->
phases through `ExperimentPlan`. Phase 1/3/4 specs depend on the previous phase's winner, so
run the cells in order; re-running the whole notebook after a crash is the resume path. Keep
the per-mode sweep loops commented unless the run is meant to include them.
`multirun_wip.ipynb` is the BI-RADS label ablation and does not change `src/`.

Keep notebook configuration and source defaults synchronized when adding a mode or changing bag,
patch, checkpoint, or dataset behavior. Update the workspace `README.md` and `src/README.md`
when the enabled modes, canvases, checkpoint metric, or logged Comet sweep change.
