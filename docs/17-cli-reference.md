# 17 — CLI and Configuration Reference

This document is a handover-grade reference for operating this engine. It covers every command, every option, every default, and every environment variable, each pulled directly from [`rde/cli.py`](../rde/cli.py) and [`rde/config.py`](../rde/config.py) rather than reconstructed from memory. If this document and the source code ever disagree, the source code is correct and this document has drifted from it.

This document is written for someone who did not build this system and needs to run it, extend it, or debug it today. A reader who has never run the engine should begin with Section 1; a reader who already has, and needs a specific command's exact flags, should proceed directly to Section 5.

## 1. First Run, in Order

The following sequence of commands takes a fresh checkout to a working, evaluated dataset.

```bash
cd episodic-data-engine
python3.11 -m venv ../.venv && ../.venv/bin/pip install -e ".[dev]"

# every data-touching command should run under the memory cap on shared/WSL
# hosts -- see §3. The examples below assume the venv is activated:
source ../.venv/bin/activate

rde info                                    # confirm paths, before anything writes
rde run --config configs/so100_pickplace.json
rde obs runs so100_pickplace                # confirm it actually ran
rde explore overview so100_pickplace training/v1
```

`rde run` alone takes a dataset from nothing to a trained and evaluated `training/v1`: it downloads, ingests, validates, transforms, re-validates, builds, evaluates, and runs feedback, all in a single call. Every other command described in this document either performs a single stage of that pipeline in isolation, or interrogates what a prior run produced.

## 2. Command Tree

The engine exposes 35 commands, organized into the hierarchy shown below.

```
rde
├── ingest              convert a source dataset into a canonical version
├── transform            apply the standard transformation pipeline
├── build                 build a training version with splits + contract
├── run                    the whole pipeline from a config file
├── info                   show configuration and storage layout
├── source
│   ├── download          fetch a dataset into the source cache
│   └── inspect            profile a source dataset without ingesting it
├── qc
│   ├── list-checks       list every registered validation check
│   ├── run                 validate a version
│   ├── cache                inspect / clear the incremental-validation cache
│   └── findings            show recorded QC findings
├── explore
│   ├── overview           episodes, frames, tasks, modalities, storage
│   ├── tasks                task distribution
│   ├── modalities           available modalities and coverage
│   ├── durations             episode duration distribution
│   ├── dims                   state/action dimensionality + per-dim stats
│   ├── frequency                measured vs declared sampling frequency
│   ├── failed                     episodes flagged by QC, with reasons
│   ├── missing                      episodes with missing/discontinuous frames
│   ├── examples                       example episodes for one task
│   └── sql                              arbitrary SQL over the frame table
├── version
│   ├── list                list versions
│   ├── lineage                provenance chain that produced a version
│   └── trace                    every episode traced back to its raw ancestor
├── model
│   ├── list                list registered models and their adapters
│   ├── evaluate              evaluate a model against a version
│   └── feedback                 run inference, detect failures, nominate episodes
└── obs
    ├── runs                 recent pipeline runs
    ├── stages                  stage timings across runs
    ├── quality                   QC executions over time
    ├── metrics                     inspect the exported Prometheus textfile
    ├── health                        per-episode health, worst first
    ├── drift                           compare two versions' distributions
    ├── contract                          infer / evaluate a release contract
    └── video                                video health without full decode
```

Every one of these 35 commands accepts `--help`, which renders the same information Typer would print at the terminal. This document adds what `--help` does not: the reasoning behind each default, and worked examples run against the real datasets in this repository.

## 3. Running Safely — the Memory Cap

Every command that touches data should be run under `scripts/capped` rather than invoked directly. The engine is designed to stream and stay within a bounded memory footprint, but "designed to" is not the same guarantee as "is guaranteed to": a bug in a new adapter, or an unexpected dataset shape, can still cause memory usage to balloon, and on WSL specifically, an out-of-memory condition can bring down the entire virtual machine along with the offending process.

```bash
scripts/capped rde run --config configs/so100_pickplace.json
RDE_MEM_MAX=3G scripts/capped python examples/03_resource_report.py
```

Table 1 lists the two environment variables this wrapper reads.

**Table 1. `scripts/capped` environment variables.**

| Variable | Default | Meaning |
|---|---|---|
| `RDE_MEM_MAX` | `6G` | Hard ceiling; the process is sent SIGKILL (exit code 137) if it exceeds this limit |
| `RDE_SWAP_MAX` | `1G` | Swap allowance permitted on top of `RDE_MEM_MAX` |

On a host where `systemd-run` is unavailable (any non-systemd host), the wrapper falls back to running the command uncapped, with a warning printed to that effect. Every `make` target that touches data is already wrapped in this mechanism (see Section 14), so `make pipeline` is already safe to run directly; a bare `rde run` invocation is not.

## 4. Environment Variables

The three variables in Table 2 are read directly by `Settings.from_env()` in [`rde/config.py`](../rde/config.py), and are the only three the engine itself consults. The two variables belonging to `scripts/capped` are listed separately, in Table 1 above.

**Table 2. Engine-level environment variables.**

| Variable | Default | Effect |
|---|---|---|
| `RDE_DATA_ROOT` | `~/VLAExperiments/rde_data` | The location where all data lives, kept outside the source tree by design. Setting this once relocates the entire engine's storage. |
| `RDE_ROW_GROUP_SIZE` | `2048` | Parquet rows per row group. Smaller groups improve predicate-pushdown selectivity, at the cost of additional per-group overhead. |
| `RDE_BATCH_SIZE` | `1024` | Frames held in memory per batch during validation and transformation. This is the first setting to lower if a stage runs out of memory. |
| `HF_TOKEN` | Unset | A Hugging Face Hub token, required by `source download` for gated or private repositories. |

```bash
RDE_DATA_ROOT=/mnt/big/rde RDE_BATCH_SIZE=512 rde run --config configs/x.json
```

`rde info` prints the resolved values of these variables for the current process. It is worth running first whenever the engine's behavior seems inconsistent with expectation.

## 5. Top-Level Commands

### 5.1 `rde ingest`

This command converts a source dataset already present on disk — for example, one obtained via `rde source download` — into a canonical raw version.

```
rde ingest PATH --dataset-id/-d TEXT [OPTIONS]
```

**Table 3. `rde ingest` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `path` | Path (positional) | — | Source dataset root |
| `--dataset-id`, `-d` | Text | — | Required |
| `--version`, `-v` | Text | `raw/v1` | Output version |
| `--source-format` | Text | `lerobot` | Adapter to use; only `lerobot` is registered today |
| `--source-uri` | Text | None | Recorded provenance URI, used if `path` is not itself self-describing |

```bash
rde ingest ~/VLAExperiments/rde_data/source_cache/lerobot__svla_so100_pickplace \
  -d so100_pickplace -v raw/v1
```

### 5.2 `rde transform`

This command applies the standard transformation pipeline: quality filtering, timestamp rebasing, per-modality normalization, and an optional deferred image resize.

```
rde transform DATASET_ID [INPUT_VERSION] [OUTPUT_VERSION] [OPTIONS]
```

**Table 4. `rde transform` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `input_version` | Positional | `raw/v1` | |
| `output_version` | Positional | `clean/v1` | |
| `--normalize` | Comma-list | `state.joint,action.joint` | Modalities to fit mean/std over; an empty string skips normalization |
| `--scheme` | Text | `mean_std` | Normalization scheme passed to `NormalizeVector` |
| `--image-size` | `HxW` or empty | `224x224` | A deferred resize, recorded in the manifest and applied at read time, never materialized; pass `""` to disable it |
| `--keep-quality` | Comma-list | `passed` | `QualityStatus` values to retain — for example, `passed,warning` to also keep warned episodes |

```bash
rde transform so100_pickplace raw/v1 clean/v1 \
  --normalize state.joint,action.joint --keep-quality passed,warning
```

### 5.3 `rde build`

This command builds a training version: episode-level splits, together with the training contract the model layer reads.

```
rde build DATASET_ID [INPUT_VERSION] [OUTPUT_VERSION] [OPTIONS]
```

**Table 5. `rde build` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `input_version` | Positional | `clean/v1` | |
| `output_version` | Positional | `training/v1` | |
| `--train` | Float | `0.8` | |
| `--val` | Float | `0.1` | |
| `--test` | Float | `0.1` | Must sum to 1.0 together with `--train` and `--val` |
| `--seed` | Int | `42` | Split-hash seed |
| `--split-mode` | `quantile` \| `independent` | `quantile` | `quantile` produces exact split sizes, which matters at small episode counts; `independent` keeps an episode's split assignment stable as the dataset grows — see the discussion of `assign_splits` in [11-code-map.md](11-code-map.md) |

```bash
rde build so100_pickplace clean/v1 training/v1 --train 0.8 --val 0.1 --test 0.1
```

### 5.4 `rde run`

This command runs the whole pipeline, either from a JSON configuration file or from command-line flags.

```
rde run [--config PATH] [OPTIONS]
```

**Table 6. `rde run` options.**

| Option | Type | Default | Notes |
|---|---|---|---|
| `--config` | Path | None | A `PipelineConfig` JSON file — see Section 12. If omitted, the configuration is built from the flags below. |
| `--dataset-id` | Text | `so100_pickplace` | Used only when `--config` is omitted |
| `--repo` | Text | `lerobot/svla_so100_pickplace` | Used only when `--config` is omitted |
| `--skip-download` | Flag | `false` | Skips the download stage; use this when `source_path` in the configuration already points at ingested data |

```bash
rde run --config configs/so100_pickplace.json
rde run --config configs/so100_pickplace.json --skip-download
rde run --dataset-id my_dataset --repo lerobot/my_dataset   # no config file
```

This command writes a record to `rde_data/runs/<dataset_id>-<UTC timestamp>.json`. This file is the record to retain for reproducing a run: its embedded configuration passes directly back into `PipelineConfig.from_file`.

### 5.5 `rde info`

This command takes no arguments. It prints resolved paths and settings — `software_version`, all six path roots, `row_group_size`, `batch_size`, and `video_cache_frames` — and it is always safe to run, since it touches nothing on disk.

## 6. `rde source` — Acquiring Data

### 6.1 `rde source download`

```
rde source download REPO_ID [OPTIONS]
```

**Table 7. `rde source download` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `repo_id` | Positional | — | A Hugging Face dataset repository identifier, for example `lerobot/svla_so100_pickplace` |
| `--revision` | Text | `main` | |
| `--metadata-only` | Flag | `false` | Fetches only `meta/*` and `*.md`/`*.json` files, skipping the multi-hundred-megabyte video payload; useful for inspecting schema and episode structure before committing to a full download |

```bash
rde source download lerobot/svla_so100_pickplace
rde source download lerobot/aloha_static_coffee --metadata-only
```

This command writes into `$RDE_DATA_ROOT/source_cache/<org>__<dataset>/`, a location never mutated afterward. It requires `HF_TOKEN` only for gated or private repositories.

### 6.2 `rde source inspect`

This command profiles a source dataset without ingesting it: no canonical version is created, and nothing is written to the catalog.

```
rde source inspect PATH [OPTIONS]
```

**Table 8. `rde source inspect` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `path` | Positional | — | Path to an already-downloaded source dataset root |
| `--source-format` | Text | `lerobot` | |
| `--dataset-id` | Text | `inspect` | Cosmetic only — used in the printed profile; nothing is persisted under this identifier |

```bash
rde source inspect ~/VLAExperiments/rde_data/source_cache/lerobot__svla_so100_pickplace
```

This command prints the source profile — episode and frame counts, fps, and detected anomalies — together with the canonical modality mapping table: every `source_key → canonical name` rule the adapter would apply on ingest, so that the mapping can be verified before committing to it.

## 7. `rde qc` — Validation

### 7.1 `rde qc list-checks`

This command takes no arguments. It lists all 24 registered checks, with their scope, severity, priority, and description.

```bash
rde qc list-checks
```

### 7.2 `rde qc run`

```
rde qc run DATASET_ID [VERSION] [OPTIONS]
```

**Table 9. `rde qc run` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `version` | Positional | `raw/v1` | |
| `--checks` | Comma-list | None (all) | Restricts execution to the named checks, for example `structural.frame_schema,temporal.gaps` |
| `--deep-video` | Flag | `false` | Fully decodes video for the integrity check instead of probing both ends; slow but thorough |
| `--max-warnings` | Int | None | Episodes with more than this many non-blocking findings are quarantined rather than passed |
| `--incremental` | Flag | `false` | Reuses a previous validation when the version, its blobs, and the check suite are all unchanged — measured at 35× faster on the SO-100 dataset (2.73 s to 0.08 s). This is opt-in, since a stale QC result is a correctness failure in the one component whose entire job is correctness. |

```bash
rde qc run so100_pickplace raw/v1
rde qc run so100_pickplace raw/v1 --incremental        # cache-aware
rde qc run so100_pickplace raw/v1 --checks temporal.gaps,structural.frame_schema
rde qc run aloha_coffee raw/v1 --deep-video --max-warnings 5
```

### 7.3 `rde qc cache`

This command inspects or clears the incremental-validation cache backing `--incremental`.

```
rde qc cache [DATASET_ID] [--clear]
```

**Table 10. `rde qc cache` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | None (all datasets) | |
| `--clear` | Flag | `false` | Invalidates cached entries; requires `dataset_id` to be given |

```bash
rde qc cache so100_pickplace              # show hit counts, entry sizes
rde qc cache so100_pickplace --clear      # invalidate after editing a check
```

### 7.4 `rde qc findings`

```
rde qc findings DATASET_ID [VERSION] [OPTIONS]
```

**Table 11. `rde qc findings` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `version` | Positional | `raw/v1` | |
| `--episode` | Int | None | Filters to one episode |
| `--check` | Text | None | Filters to one check name |
| `--severity` | Text | None | Filters to one severity (`info`, `warning`, `error`, or `critical`) |

```bash
rde qc findings so100_pickplace raw/v1 --episode 9
rde qc findings aloha_coffee raw/v1 --severity warning
```

## 8. `rde explore` — Querying a Version

All ten `explore` subcommands take `dataset_id` as a positional argument. Most default `version` to `training/v1`; two default it to `raw/v1` instead, noted in Table 12, because they exist specifically to investigate raw-ingest problems before anything is built on top of them.

**Table 12. `rde explore` subcommands.**

| Command | Signature | Version default | Purpose |
|---|---|---|---|
| `overview` | `DATASET_ID [VERSION]` | `training/v1` | Episodes, frames, tasks, modalities, and storage footprint |
| `tasks` | `DATASET_ID [VERSION]` | `training/v1` | Task distribution |
| `modalities` | `DATASET_ID [VERSION]` | `training/v1` | Every modality, with coverage and normalization status |
| `durations` | `DATASET_ID [VERSION]` | `training/v1` | Episode duration distribution and histogram |
| `dims` | `DATASET_ID [VERSION]` | `training/v1` | Per-dimension state and action statistics |
| `frequency` | `DATASET_ID [VERSION]` | `training/v1` | Measured versus declared sampling frequency |
| `failed` | `DATASET_ID [VERSION]` | `raw/v1` | Episodes flagged by QC, with reasons, printed hierarchically |
| `missing` | `DATASET_ID [VERSION]` | `raw/v1` | Episodes with missing or discontinuous frames |
| `examples` | `DATASET_ID TASK_ID [--version] [--limit]` | `training/v1` | Example episodes for one task identifier |
| `sql` | `DATASET_ID QUERY [--version]` | `training/v1` | Arbitrary SQL against the `frames` view, executed via DuckDB |

```bash
rde explore overview so100_pickplace training/v1
rde explore durations aloha_coffee raw/v1
rde explore dims so100_pickplace training/v1
rde explore failed so100_pickplace raw/v1
rde explore examples so100_pickplace 0 --limit 3
rde explore sql so100_pickplace "SELECT episode_id, count(*) FROM frames GROUP BY 1 LIMIT 5"
```

The `sql` subcommand is the escape hatch of this group, exposing anything DuckDB can express against the frame table. Dotted modality columns must be quoted — `"state.joint[1]"` rather than `state.joint[1]` — since DuckDB would otherwise parse the dot as a `table.column` reference.

## 9. `rde version` — Lineage

**Table 13. `rde version` subcommands.**

| Command | Signature | Purpose |
|---|---|---|
| `list` | `[DATASET_ID]` | Lists versions; all datasets if `dataset_id` is omitted |
| `lineage` | `DATASET_ID [VERSION="training/v1"]` | The full provenance chain: every transform edge that produced this version, with run identifier, code fingerprint, and software version |
| `trace` | `DATASET_ID [VERSION="training/v1"]` | Every episode of this version, traced back to its raw ancestor (the first 50 are shown) |

```bash
rde version list
rde version list so100_pickplace
rde version lineage so100_pickplace training/v1
rde version trace so100_pickplace training/v1
```

## 10. `rde model` — Evaluation and Feedback

### 10.1 `rde model list`

This command takes no arguments. It prints the model registry, given in Table 14.

**Table 14. Model registry.**

| Name | Class | Adapter |
|---|---|---|
| `mean_action` | `MeanActionPolicy` | `StateAdapter` |
| `nearest_neighbor` | `NearestNeighborPolicy` | `StateAdapter` |
| `smolvla` | `SmolVLAPolicy` | `VisionLanguageAdapter` |

### 10.2 `rde model evaluate`

```
rde model evaluate DATASET_ID [VERSION] [OPTIONS]
```

**Table 15. `rde model evaluate` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `version` | Positional | `training/v1` | |
| `--model` | Text | `nearest_neighbor` | Must be a key in `MODEL_REGISTRY` |
| `--baseline` | Text | `mean_action` | Comparison model; pass `--baseline ""` to skip the comparison |
| `--split` | Text | `test` | |
| `--max-frames` | Int | `2000` | Cap on evaluated frames; omit, or pass `None`, to evaluate the whole split |

```bash
rde model evaluate so100_pickplace training/v1 --model nearest_neighbor
rde model evaluate so100_pickplace training/v1 --model smolvla --baseline ""
```

This command fits the model on the `train` split internally, evaluates it on `--split`, and prints the summary together with a per-dimension MAE breakdown.

### 10.3 `rde model feedback`

This command runs inference, detects failures against relative, percentile-based thresholds, and nominates episodes for the next training round.

```
rde model feedback DATASET_ID [VERSION] [OPTIONS]
```

**Table 16. `rde model feedback` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `version` | Positional | `training/v1` | |
| `--model` | Text | `nearest_neighbor` | |
| `--split` | Text | `test` | |
| `--max-frames` | Int | `2000` | |
| `--quarantine` | Flag | `false` | Actually marks nominated episodes for review in the catalog, rather than only reporting them |

```bash
rde model feedback so100_pickplace training/v1
rde model feedback so100_pickplace training/v1 --quarantine
```

This command prints the failure report, the top ten nominated episodes with their priority and recommendation, and the `episode_weights` a subsequent `build` command would use to oversample them — the point at which the feedback flywheel closes.

## 11. `rde obs` — Observability

The full reference for what each of these commands reads, and why, is given in [15-metrics-catalog.md](15-metrics-catalog.md); this section is confined to command syntax.

### 11.1 `rde obs runs`

```
rde obs runs DATASET_ID [--limit N=20]
```

This command lists recent pipeline runs, newest first, showing status, duration, and configuration hash.

### 11.2 `rde obs stages`

```
rde obs stages DATASET_ID [--stage NAME] [--limit N=30]
```

This command lists stage timings across runs. Omitting `--stage` shows all stages; passing one, such as `validate`, directly answers the question "has this stage gotten slower?"

### 11.3 `rde obs quality`

```
rde obs quality DATASET_ID [--version V] [--limit N=20]
```

This command lists QC executions over time. Unlike `qc findings`, which shows only current findings — findings are replaced on each re-validation — this command shows the trend across executions, because execution headers accumulate even though the findings underneath them do not.

### 11.4 `rde obs metrics`

```
rde obs metrics [--show] [--max-age-minutes N=60.0]
```

This command inspects the exported Prometheus textfile at `$RDE_DATA_ROOT/observability/metrics/rde.prom`. It reports series count, instrument count, and whether the export is stale, where staleness is defined as a file modification time older than `--max-age-minutes`, with a missing file treated as stale by definition. Passing `--show` prints the full exposition text.

```bash
rde obs metrics
rde obs metrics --show
rde obs metrics --max-age-minutes 15
```

### 11.5 `rde obs health`

```
rde obs health DATASET_ID [VERSION="raw/v1"] [--limit N=15]
```

This command shows per-episode health drawn from the analytical event rows, ranked worst first by combined temporal, numeric, and alignment quality. It is scoped to the most recent QC execution for the given version, so re-validating refreshes the view automatically.

### 11.6 `rde obs drift`

```
rde obs drift DATASET_ID BASELINE CANDIDATE [--statistic S="psi,mean_shift"]
```

**Table 17. `rde obs drift` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `baseline` | Positional | — | Version treated as the reference distribution |
| `candidate` | Positional | — | Version compared against the baseline |
| `--statistic` | Comma-list | `psi,mean_shift` | Which statistics to compute per feature |

```bash
rde obs drift so100_pickplace raw/v90 raw/v91
```

This command compares episode-level feature distributions, never frame-level ones — see [13-observability-industry-research.md](13-observability-industry-research.md), Section 5. Each reported row includes its effective sample size beside its value, and a comparison judged underpowered is marked as such rather than silently treated as statistically significant.

### 11.7 `rde obs contract`

```
rde obs contract DATASET_ID BASELINE [--candidate V] [--out PATH]
```

**Table 18. `rde obs contract` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `baseline` | Positional | — | Version the contract is inferred from |
| `--candidate` | Text | None | Version checked against the inferred contract; omit to only infer and print the contract |
| `--out` | Path | None | Writes the inferred contract as JSON to this path |

```bash
rde obs contract so100_pickplace raw/v1                              # infer only
rde obs contract so100_pickplace raw/v1 --candidate clean/v1         # infer + check
rde obs contract so100_pickplace raw/v1 --out contracts/so100.json   # infer + save
```

This command's exit code doubles as the release gate: `0` if released, `2` if blocked, making it safe to use directly in continuous integration, for example as `rde obs contract ... --candidate ... || exit 1`. Contracts are evaluated per stage: a contract inferred from `raw/v1` will reject `clean/v1` if a transform resized images, since that constitutes a real, if intended, schema change. A contract should be inferred from the stage actually intended for release.

### 11.8 `rde obs video`

```
rde obs video DATASET_ID [VERSION="raw/v1"] [--tier 0|1] [--limit N=10]
```

**Table 19. `rde obs video` arguments and options.**

| Argument/option | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | Positional | — | Required |
| `version` | Positional | `raw/v1` | |
| `--tier` | `0` \| `1` | `0` | Tier 0 checks container headers only (approximately 2 ms per file); Tier 1 performs a sparse decode of 5 frames per clip (approximately 90 ms) |
| `--limit` | Int | `10` | Number of episode-camera references sampled at Tier 1 only; Tier 0 always covers every distinct file |

```bash
rde obs video so100_pickplace raw/v1                    # tier 0, all files
rde obs video so100_pickplace raw/v1 --tier 1 --limit 20
```

Tier 0 exits with code `1` if any container is unreadable, making it usable as a fast continuous-integration gate on newly ingested video, ahead of running full QC.

## 12. `PipelineConfig` — the `rde run --config` Schema

This section documents every field of `PipelineConfig`, its type, and its default value. The source of truth for this schema is [`rde/pipeline.py`](../rde/pipeline.py).

```json
{
  "dataset_id": "so100_pickplace",
  "source_repo": "lerobot/svla_so100_pickplace",
  "source_path": null,
  "source_format": "lerobot",

  "raw_version": "raw/v1",
  "clean_version": "clean/v1",
  "training_version": "training/v1",

  "normalize": ["state.joint", "action.joint"],
  "normalize_scheme": "mean_std",
  "image_size": [224, 224],
  "keep_quality": ["passed"],

  "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
  "split_seed": 42,
  "split_mode": "quantile",

  "model": "nearest_neighbor",
  "baseline_model": "mean_action",
  "eval_split": "test",
  "eval_max_frames": 2000,

  "deep_video_check": false
}
```

**Table 20. `PipelineConfig` fields.**

| Field | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | string | — | Required |
| `source_repo` | string \| null | `null` | A Hugging Face repository identifier; set this or `source_path`, never both |
| `source_path` | string \| null | `null` | A local path to an already-downloaded source; setting this implies the same behavior as `--skip-download` |
| `source_format` | string | `"lerobot"` | Adapter name |
| `raw_version` / `clean_version` / `training_version` | string | See the example above | Output version names for each pipeline stage |
| `normalize` | list[string] | `["state.joint","action.joint"]` | Modalities normalized during the `transform` stage |
| `normalize_scheme` | string | `"mean_std"` | |
| `image_size` | `[H,W]` \| null | `[224,224]` | A deferred resize; `null` disables it |
| `keep_quality` | list[string] | `["passed"]` | `QualityStatus` values kept by the `transform` stage's quality filter |
| `splits` | object | `{"train":0.8,"val":0.1,"test":0.1}` | Must sum to 1.0 |
| `split_seed` | int | `42` | |
| `split_mode` | `"quantile"` \| `"independent"` | `"quantile"` | See Section 5.3, `rde build` |
| `model` | string | `"nearest_neighbor"` | A key into `MODEL_REGISTRY` |
| `baseline_model` | string \| null | `"mean_action"` | `null` skips the comparison performed during the `evaluate` stage |
| `eval_split` | string | `"test"` | |
| `eval_max_frames` | int \| null | `2000` | |
| `deep_video_check` | bool | `false` | Escalates `structural.video_integrity` to a full decode on both QC passes |

Two reference configurations ship in [`configs/`](../configs/): `so100_pickplace.json` and `aloha_static_coffee.json`. The two differ only in `dataset_id`, `source_repo`, and the addition of `force.effort` to `normalize` in the latter.

## 13. Model Registry, Transformation Operators, and Modality Kinds

This section is a quick reference for the names accepted by `--model`, the transformation classes available in Python (importable beyond the fixed pipeline `rde transform` composes), and the modality vocabulary.

Table 21 lists the model registry, defined in [`rde/model_adapters/models.py`](../rde/model_adapters/models.py).

**Table 21. Model registry.**

| Name | Class | Adapter | Notes |
|---|---|---|---|
| `mean_action` | `MeanActionPolicy` | `StateAdapter` | Predicts the training-split mean action — the floor every other model must beat |
| `nearest_neighbor` | `NearestNeighborPolicy` | `StateAdapter` | A retrieval baseline, using Gram-matrix distance computation and bounded memory |
| `smolvla` | `SmolVLAPolicy` | `VisionLanguageAdapter` | A real vision-language-action model; requires the separate `.venv-vla` environment described in [07-model-integration.md](07-model-integration.md) |

Table 22 lists the transformation operators available in `rde.transformations`.

**Table 22. Transformation operators.**

| Class | Shape | Purpose |
|---|---|---|
| `NormalizeVector(modality, scheme)` | `FrameTransform` | Fits and applies mean/std normalization |
| `TimestampNormalizer()` | `FrameTransform` | Rebases timestamps to be episode-relative |
| `SelectFeatures(keep)` | `FrameTransform` | Drops columns not named in `keep` |
| `ResizeImages(h, w)` | `FrameTransform` | Records a deferred resize in the manifest |
| `NormalizeImages(mean, std)` | `FrameTransform` | Records a deferred image normalization |
| `DropNonFiniteFrames(modality)` | `FrameFilter` | Drops rows containing NaN or Inf values |
| `SubsampleFrames(stride)` | `FrameFilter` | Keeps every Nth frame |
| `QualityFilter(statuses)` | `EpisodeFilter` | Keeps only episodes at the given `QualityStatus` values |
| `LengthFilter(min_frames, max_frames)` | `EpisodeFilter` | Drops episodes outside a given frame-count range |
| `TaskFilter(task_ids)` | `EpisodeFilter` | Keeps only the named tasks |

The modality kinds available in `ModalityKind`, an open registry described in [02-canonical-schema.md](02-canonical-schema.md), are: `rgb`, `depth`, `robot_state`, `robot_action`, `hand_state`, `tactile`, `force_torque`, `audio`, `point_cloud`, `language`, and `other`.

The storage planes available in `StoragePlane` are `inline` (a numeric column in the frame Parquet table) and `external` (the blob store, referenced by identifier).

## 14. `Makefile` Targets

Every data-touching target listed in Table 23 already runs under `scripts/capped`; these targets are preferred over calling `rde` directly for anything beyond quick interrogation. Three variables govern their behavior: `DATASET` (default `so100_pickplace`), `TIER` (default `0`), and `CONFIG` (default `configs/so100_pickplace.json`).

**Table 23. `Makefile` targets.**

| Target | Runs | Override |
|---|---|---|
| `make install` | Editable install plus dev extras | |
| `make test` | Full test suite | |
| `make test-cov` | Test suite with a coverage report | |
| `make pipeline` | `rde run --config $(CONFIG)` on SO-100 | `CONFIG=path/to.json` |
| `make pipeline-aloha` | `rde run --config configs/aloha_static_coffee.json` | |
| `make inspect` | `rde source inspect` on the source cache | |
| `make faults` | The fault-injection proof, confirming QC catches every defect class | |
| `make resources` | The resource usage report underlying [09-resource-usage.md](09-resource-usage.md) | |
| `make models` | The model interchangeability demonstration | |
| `make lineage` | Answers the two provenance questions from [06-versioning-lineage.md](06-versioning-lineage.md) | |
| `make flywheel` | Closes the deployment feedback loop | |
| `make compare` | Compares the engine across both proof datasets | |
| `make observability` | The two end-to-end failure investigations (sensor and drift) | |
| `make cardinality` | Measures the cost of using `episode_id` as a metric label | |
| `make obs-runs` | `rde obs runs $(DATASET)` | `DATASET=aloha_coffee` |
| `make obs-health` | `rde obs health $(DATASET) raw/v1` | |
| `make obs-video` | `rde obs video $(DATASET) raw/v1 --tier $(TIER)` | `TIER=1` |
| `make obs-contract` | `rde obs contract $(DATASET) raw/v1` | |
| `make obs-metrics` | `rde obs metrics` | |
| `make checks` | `rde qc list-checks` | |
| `make versions` | `rde version list` | |
| `make clean-derived` | Destructive: drops `catalog.db`, `tables/`, `artifacts/`, and `runs/`, while keeping `source_cache/` and `blobs/` | |

```bash
make pipeline-aloha
make obs-video TIER=1 DATASET=aloha_coffee
make pipeline CONFIG=configs/my_dataset.json
```

## 15. Common Workflows

This section presents five workflows as short numbered command sequences, each addressing a common operational task.

**Bringing up a new dataset end to end.**

```bash
rde source download lerobot/<repo>
scripts/capped rde run --config configs/<new>.json
rde obs runs <dataset_id>
```

**Re-validating after editing a check, without re-ingesting.**

```bash
rde qc cache <dataset_id> --clear      # the suite changed; don't trust the cache
rde qc run <dataset_id> raw/v1
```

**Investigating why an episode was rejected.**

```bash
rde explore failed <dataset_id> raw/v1
rde qc findings <dataset_id> raw/v1 --episode <N>
rde obs health <dataset_id> raw/v1
```

**Deciding whether a new version is safe to release.**

```bash
rde obs contract <dataset_id> raw/v1 --candidate raw/v2
echo $?   # 0 = release, 2 = blocked
```

**Swapping the model without touching data.**

```bash
rde model evaluate <dataset_id> training/v1 --model mean_action
rde model evaluate <dataset_id> training/v1 --model nearest_neighbor
rde model evaluate <dataset_id> training/v1 --model smolvla
```

**Checking pipeline health before a demonstration.**

```bash
rde obs runs <dataset_id> --limit 5
rde obs metrics
rde obs stages <dataset_id>
```

## 16. Where to Go Next

Table 24 points to further documentation depending on what a reader needs next.

**Table 24. Further reading, by need.**

| Need | Document |
|---|---|
| What a command's output means | [15-metrics-catalog.md](15-metrics-catalog.md), [05-qc-architecture.md](05-qc-architecture.md) |
| Which file implements a command's logic | [11-code-map.md](11-code-map.md) |
| Why a default is what it is | [10-design-decisions.md](10-design-decisions.md) |
| A worked end-to-end trace | [08-end-to-end-run.md](08-end-to-end-run.md) |
| The data currently in the catalog | `rde explore overview`, `rde explore dims`, `rde explore failed` ([§8](#8-rde-explore--querying-a-version) above) |
