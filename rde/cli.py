"""Command-line interface.

Exposes each pipeline stage plus an exploration surface, so the engine can be
driven and interrogated without writing Python.  Grouped by stage::

    rde source     download / inspect an external dataset
    rde ingest     convert a source into a canonical raw version
    rde qc         run validation, list checks, show findings
    rde transform  apply operators, produce a derived version
    rde build      produce a training version with splits
    rde explore    query any version
    rde version    list versions, show lineage
    rde model      evaluate a model, run the feedback loop
    rde run        the whole pipeline from a config file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.markup import escape
from rich.table import Table

from .catalog import Catalog
from .config import get_settings
from .schemas.core import QualityStatus

app = typer.Typer(help="Episodic Data Engine", no_args_is_help=True, add_completion=False)
source_app = typer.Typer(help="Acquire and inspect source datasets", no_args_is_help=True)
qc_app = typer.Typer(help="Validation and quality control", no_args_is_help=True)
explore_app = typer.Typer(help="Query datasets", no_args_is_help=True)
version_app = typer.Typer(help="Versions and lineage", no_args_is_help=True)
model_app = typer.Typer(help="Model evaluation and feedback", no_args_is_help=True)
obs_app = typer.Typer(help="Observability: run history, metrics, data health", no_args_is_help=True)
app.add_typer(source_app, name="source")
app.add_typer(qc_app, name="qc")
app.add_typer(explore_app, name="explore")
app.add_typer(version_app, name="version")
app.add_typer(model_app, name="model")
app.add_typer(obs_app, name="obs")

console = Console()


def _print(obj, title: str | None = None):
    if title:
        console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(JSON(json.dumps(obj, indent=2, default=str)))


def _table(rows: list[dict], title: str, columns: list[str] | None = None):
    if not rows:
        console.print(f"[yellow]{title}: no rows[/yellow]")
        return
    cols = columns or list(rows[0])
    t = Table(title=title, show_lines=False, header_style="bold cyan")
    for c in cols:
        t.add_column(c, overflow="fold")
    for r in rows:
        t.add_row(*[str(r.get(c, "")) for c in cols])
    console.print(t)


# -- source -----------------------------------------------------------------
@source_app.command("download")
def source_download(
    repo_id: str = typer.Argument(..., help="Hugging Face dataset repo id"),
    revision: str = typer.Option("main"),
    metadata_only: bool = typer.Option(False, help="Fetch meta/ only, skip video payload"),
):
    """Download a dataset into the source cache (outside the code tree)."""
    from .ingestion import download_hf_dataset

    patterns = ["meta/*", "meta/**/*", "*.md", "*.json"] if metadata_only else None
    loc = download_hf_dataset(repo_id, revision, allow_patterns=patterns)
    _print({**loc.to_json(), "size_mb": round(loc.size_bytes / 1e6, 1)}, "Downloaded")


@source_app.command("inspect")
def source_inspect(
    path: Path = typer.Argument(..., help="Path to a source dataset root"),
    source_format: str = typer.Option("lerobot"),
    dataset_id: str = typer.Option("inspect"),
):
    """Inspect a source dataset without ingesting it."""
    from .ingestion import ADAPTERS

    adapter = ADAPTERS[source_format](path, dataset_id)
    profile = adapter.inspect()
    _print(profile.summary(), f"Source profile: {path}")
    rows = [
        {"source_key": m.source_key, "canonical": m.name, "kind": m.kind.value,
         "plane": m.plane.value, "dtype": m.dtype, "shape": str(list(m.shape))}
        for m in adapter._modalities()
    ]
    _table(rows, "Canonical modality mapping")


# -- ingest -----------------------------------------------------------------
@app.command("ingest")
def cmd_ingest(
    path: Path = typer.Argument(..., help="Source dataset root"),
    dataset_id: str = typer.Option(..., "--dataset-id", "-d"),
    version: str = typer.Option("raw/v1", "--version", "-v"),
    source_format: str = typer.Option("lerobot"),
    source_uri: Optional[str] = typer.Option(None),
):
    """Convert a source dataset into a canonical version."""
    from .ingestion import ingest

    res = ingest(path, dataset_id, source_format, version, source_uri=source_uri)
    _print(res.summary(), f"Ingested {dataset_id}@{version}")


# -- qc ---------------------------------------------------------------------
@qc_app.command("list-checks")
def qc_list_checks():
    """List every registered validation check."""
    from .validation import default_registry

    rows = [
        {"name": c.name, "scope": c.scope.value, "severity": c.severity.value,
         "priority": c.priority, "description": c.description}
        for c in sorted(default_registry(), key=lambda c: (c.scope.value, c.priority, c.name))
    ]
    _table(rows, f"Registered checks ({len(rows)})",
           ["name", "scope", "severity", "description"])


@qc_app.command("run")
def qc_run(
    dataset_id: str = typer.Argument(...),
    version: str = typer.Argument("raw/v1"),
    checks: Optional[str] = typer.Option(None, help="Comma-separated check names"),
    deep_video: bool = typer.Option(False, help="Fully decode video (slow, thorough)"),
    max_warnings: Optional[int] = typer.Option(None, help="Quarantine above this many warnings"),
    incremental: bool = typer.Option(
        False, help="Reuse a previous validation when nothing relevant changed"
    ),
):
    """Run validation over a dataset version.

    ``--incremental`` short-circuits when the version's tables, the blobs it
    references, and the check suite's source are all unchanged -- 35x faster on
    SO-100 (2.73s to 0.08s). Opt-in on purpose: a stale QC result is a
    correctness failure in the component whose entire job is correctness.
    """
    from .qc import QCEngine, QCGate
    from .validation import default_registry
    from .validation.structural import VideoIntegrityCheck

    registry = default_registry()
    if deep_video:
        registry._checks["structural.video_integrity"] = VideoIntegrityCheck(deep=True)
    eng = QCEngine(
        registry=registry, gate=QCGate(max_warnings=max_warnings),
        incremental=incremental,
    )
    res = eng.run(dataset_id, version, checks=checks.split(",") if checks else None)
    summary = res.summary()
    if incremental:
        summary["served_from_cache"] = eng.last_run_cached
    _print(summary, f"QC {dataset_id}@{version}")


@qc_app.command("cache")
def qc_cache(
    dataset_id: Optional[str] = typer.Argument(None),
    clear: bool = typer.Option(False, "--clear", help="Invalidate cached results"),
):
    """Inspect or clear the incremental-validation cache.

    A cache whose hit rate nobody can see is a cache nobody should trust.
    """
    from .qc.cache import QCCache

    cache = QCCache(_catalog())
    if clear:
        if not dataset_id:
            console.print("[red]--clear requires a dataset_id[/red]")
            raise typer.Exit(1)
        console.print(f"[green]invalidated:[/green] {cache.invalidate(dataset_id)} entries")
        raise typer.Exit(0)
    rows = cache.stats(dataset_id)
    if not rows:
        console.print("[yellow]cache is empty[/yellow]")
        raise typer.Exit(0)
    _table(
        [{"dataset": r["dataset_id"], "version": r["version"],
          "suite": r["suite_hash"][:8], "planes": r["table_hash"][:8],
          "hits": r["hits"], "kb": round(r["bytes"] / 1024, 1),
          "cached": (r["created_at"] or "")[:19]} for r in rows],
        "Incremental validation cache",
    )


@qc_app.command("findings")
def qc_findings(
    dataset_id: str = typer.Argument(...),
    version: str = typer.Argument("raw/v1"),
    episode: Optional[int] = typer.Option(None),
    check: Optional[str] = typer.Option(None),
    severity: Optional[str] = typer.Option(None),
):
    """Show recorded QC findings."""
    cat = Catalog(get_settings().paths.catalog_db)
    rows = cat.query_qc(dataset_id, version, episode_id=episode, check_name=check,
                        severity=severity)
    _table(
        [{"episode": r["episode_id"], "check": r["check_name"], "severity": r["severity"],
          "status": r["status"], "reason": r["reason"]} for r in rows],
        f"QC findings for {dataset_id}@{version} ({len(rows)})",
    )


# -- transform / build ------------------------------------------------------
@app.command("transform")
def cmd_transform(
    dataset_id: str = typer.Argument(...),
    input_version: str = typer.Argument("raw/v1"),
    output_version: str = typer.Argument("clean/v1"),
    normalize: str = typer.Option("state.joint,action.joint", help="Modalities to normalize"),
    scheme: str = typer.Option("mean_std"),
    image_size: Optional[str] = typer.Option("224x224", help="Deferred resize, e.g. 224x224"),
    keep_quality: str = typer.Option("passed", help="Comma-separated statuses to keep"),
):
    """Apply the standard transformation pipeline."""
    from .transformations import (
        NormalizeVector, QualityFilter, ResizeImages, TimestampNormalizer, TransformPipeline,
    )

    ops = [
        QualityFilter([QualityStatus(q) for q in keep_quality.split(",")]),
        TimestampNormalizer(),
    ]
    ops += [NormalizeVector(m, scheme) for m in normalize.split(",") if m]
    if image_size:
        h, w = (int(x) for x in image_size.lower().split("x"))
        ops.append(ResizeImages(h, w))
    res = TransformPipeline(ops).run(dataset_id, input_version, output_version)
    _print(res.summary(), f"Transformed -> {output_version}")


@app.command("build")
def cmd_build(
    dataset_id: str = typer.Argument(...),
    input_version: str = typer.Argument("clean/v1"),
    output_version: str = typer.Argument("training/v1"),
    train: float = typer.Option(0.8),
    val: float = typer.Option(0.1),
    test: float = typer.Option(0.1),
    seed: int = typer.Option(42),
    split_mode: str = typer.Option("quantile", help="quantile | independent"),
):
    """Build a training dataset version with splits and a contract."""
    from .dataset_builder import DatasetBuilder

    res = DatasetBuilder().build(
        dataset_id, input_version, output_version,
        splits={"train": train, "val": val, "test": test},
        seed=seed, split_mode=split_mode,
    )
    _print(res.summary(), f"Built {output_version}")


# -- explore ----------------------------------------------------------------
def _explorer():
    from .exploration.queries import DatasetExplorer

    return DatasetExplorer()


def _catalog():
    from .catalog import Catalog

    return Catalog(get_settings().paths.catalog_db)


@explore_app.command("overview")
def explore_overview(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """Episodes, frames, tasks, modalities, storage."""
    o = _explorer().overview(dataset_id, version)
    d = o.to_dict()
    mods = d.pop("modalities")
    _print(d, f"Overview {dataset_id}@{version}")
    _table(mods, "Modalities")


@explore_app.command("tasks")
def explore_tasks(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """Task distribution."""
    _table(_explorer().task_summary(dataset_id, version), "Tasks")


@explore_app.command("modalities")
def explore_modalities(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """Available modalities and their coverage."""
    rows = _explorer().modality_summary(dataset_id, version)
    _table(rows, "Modalities",
           ["name", "kind", "plane", "dtype", "shape", "fps", "coverage", "normalized"])


@explore_app.command("durations")
def explore_durations(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """Episode duration distribution."""
    d = _explorer().duration_distribution(dataset_id, version)
    hist = d.pop("histogram", [])
    _print(d, "Duration distribution")
    _table(hist, "Histogram (seconds)")


@explore_app.command("dims")
def explore_dims(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """State/action dimensionality and per-dimension statistics."""
    for name, info in _explorer().dimensionality(dataset_id, version).items():
        _table(info.get("per_dimension", []), f"{name} ({info.get('dimensions')}-D)")


@explore_app.command("frequency")
def explore_frequency(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """Measured vs declared sampling frequency."""
    _print(_explorer().timestamp_frequency(dataset_id, version), "Sampling frequency")


@explore_app.command("failed")
def explore_failed(dataset_id: str = typer.Argument(...), version: str = typer.Argument("raw/v1")):
    """Episodes that failed or were flagged by QC, with reasons."""
    rows = _explorer().failed_episodes(dataset_id, version)
    for r in rows:
        # escape(): a bare "[warning]" is valid rich markup and would be parsed
        # as a style tag and swallowed, silently dropping the status from output.
        console.print(
            f"[bold]episode {r['episode_id']}[/bold] {escape('[' + r['status'] + ']')} "
            f"{r['num_frames']} frames, {r['duration_s']}s"
        )
        for f in r["findings"]:
            console.print(
                f"   {escape('[' + f['severity'] + ']')} {f['check']}: {escape(f['reason'])}"
            )
    if not rows:
        console.print("[green]no episodes flagged[/green]")


@explore_app.command("missing")
def explore_missing(dataset_id: str = typer.Argument(...), version: str = typer.Argument("raw/v1")):
    """Episodes with missing or discontinuous frames."""
    _table(_explorer().episodes_with_missing_frames(dataset_id, version), "Missing-frame findings")


@explore_app.command("examples")
def explore_examples(
    dataset_id: str = typer.Argument(...),
    task_id: int = typer.Argument(...),
    version: str = typer.Option("training/v1"),
    limit: int = typer.Option(5),
):
    """Show example episodes for a task."""
    _table(_explorer().examples_for_task(dataset_id, version, task_id, limit),
           f"Examples for task {task_id}")


@explore_app.command("sql")
def explore_sql(
    dataset_id: str = typer.Argument(...),
    query: str = typer.Argument(..., help="SQL over the 'frames' view"),
    version: str = typer.Option("training/v1"),
):
    """Run SQL against the frame table via DuckDB."""
    _table(_explorer().sql(dataset_id, version, query), "Query result")


# -- versions / lineage -----------------------------------------------------
@version_app.command("list")
def version_list(dataset_id: Optional[str] = typer.Argument(None)):
    """List dataset versions."""
    cat = Catalog(get_settings().paths.catalog_db)
    _table(cat.list_versions(dataset_id), "Versions",
           ["version_key", "stage", "total_episodes", "total_frames", "quality_status",
            "created_at"])


@version_app.command("lineage")
def version_lineage(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """Show the provenance chain that produced a version."""
    lin = _explorer().lineage(dataset_id, version)
    console.print(f"[bold cyan]Lineage of {dataset_id}@{version}[/bold cyan]")
    console.print(f"ancestors: {lin['ancestors']}")
    for r in lin["chain"]:
        console.print(
            f"  [bold]{r['operation']}[/bold] {r['inputs']} -> {r['output']}  "
            f"episodes {r['episodes_in']}->{r['episodes_out']}"
            + (f"  dropped={r['dropped']}" if r["dropped"] else "")
        )
        console.print(f"     run={r['run_id']} fingerprint={r['fingerprint']} "
                      f"software={r['software_version']}")
    console.print(f"contributing root episodes: {len(lin['root_episodes'])}")


@version_app.command("trace")
def version_trace(dataset_id: str = typer.Argument(...), version: str = typer.Argument("training/v1")):
    """Trace each episode of a version back to its raw ancestor."""
    cat = Catalog(get_settings().paths.catalog_db)
    traced = cat.trace_episodes_to_root(dataset_id, version)
    rows = [
        {"episode": eid, "ancestry": " <- ".join(f"{v}#{e}" for v, e in chain) or "(root)"}
        for eid, chain in sorted(traced.items())
    ]
    _table(rows[:50], f"Episode ancestry for {version} (first 50 of {len(rows)})")


# -- model ------------------------------------------------------------------
@model_app.command("list")
def model_list():
    """List available models and their adapters."""
    from .model_adapters import MODEL_REGISTRY

    _table(
        [{"model": k, "class": v[0].__name__, "adapter": v[1].__name__}
         for k, v in MODEL_REGISTRY.items()],
        "Model registry",
    )


@model_app.command("evaluate")
def model_evaluate(
    dataset_id: str = typer.Argument(...),
    version: str = typer.Argument("training/v1"),
    model: str = typer.Option("nearest_neighbor"),
    baseline: Optional[str] = typer.Option("mean_action"),
    split: str = typer.Option("test"),
    max_frames: Optional[int] = typer.Option(2000),
):
    """Evaluate a model against a dataset version."""
    from .engine import DataEngine
    from .evaluation import Evaluator
    from .model_adapters import build_model

    ev = Evaluator()
    with DataEngine().get_dataset(dataset_id, version) as view:
        m, a = build_model(model, view.contract)
        m.fit(view, split="train")
        res = ev.evaluate(view, m, a, split=split, max_frames=max_frames)
        if baseline:
            bm, ba = build_model(baseline, view.contract)
            bm.fit(view, split="train")
            bres = ev.evaluate(view, bm, ba, split=split, max_frames=max_frames, persist=False)
            ev.compare(res, bres)
    _print(res.summary(), f"Evaluation: {model}")
    _print(res.metrics.get("per_dimension_mae"), "Per-dimension MAE")


@model_app.command("feedback")
def model_feedback(
    dataset_id: str = typer.Argument(...),
    version: str = typer.Argument("training/v1"),
    model: str = typer.Option("nearest_neighbor"),
    split: str = typer.Option("test"),
    max_frames: Optional[int] = typer.Option(2000),
    quarantine: bool = typer.Option(False, help="Mark flagged episodes for review"),
):
    """Run inference, detect failures, and nominate episodes for review."""
    from .engine import DataEngine
    from .evaluation import Evaluator
    from .feedback import FeedbackLoop
    from .model_adapters import build_model

    with DataEngine().get_dataset(dataset_id, version) as view:
        m, a = build_model(model, view.contract)
        m.fit(view, split="train")
        res = Evaluator().evaluate(view, m, a, split=split, max_frames=max_frames)

    loop = FeedbackLoop()
    report = loop.detect(res)
    _print(report.summary(), "Feedback report")
    _table([r.to_dict() for r in report.reviews[:10]], "Episodes nominated for review",
           ["episode_id", "failure_frames", "total_frames_evaluated", "mean_error",
            "mean_confidence", "priority", "recommendation"])
    if quarantine and report.reviews:
        n = loop.quarantine_episodes(dataset_id, version, [r.episode_id for r in report.reviews])
        console.print(f"[yellow]quarantined {n} episodes for review[/yellow]")
    _print(loop.build_retraining_selection(report), "Retraining selection")


# -- full pipeline ----------------------------------------------------------
@app.command("run")
def cmd_run(
    config: Optional[Path] = typer.Option(None, help="Pipeline config JSON"),
    dataset_id: str = typer.Option("so100_pickplace"),
    repo: str = typer.Option("lerobot/svla_so100_pickplace"),
    skip_download: bool = typer.Option(False),
):
    """Run the complete pipeline end to end."""
    from .pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig.from_file(config) if config else PipelineConfig(
        dataset_id=dataset_id, source_repo=repo
    )
    record = Pipeline(cfg).run(skip_download=skip_download)
    _table(
        [{"stage": s["name"], "ok": s["ok"], "seconds": s["duration_s"]}
         for s in record["stages"]],
        f"Pipeline complete in {record['duration_s']}s",
    )
    console.print(f"[green]run record:[/green] {record['run_record']}")


@app.command("info")
def cmd_info():
    """Show configuration and storage layout."""
    s = get_settings()
    p = s.paths
    _print(
        {
            "software_version": s.software_version,
            "data_root": str(p.data_root),
            "source_cache": str(p.source_cache),
            "blobs": str(p.blobs),
            "tables": str(p.tables),
            "artifacts": str(p.artifacts),
            "catalog_db": str(p.catalog_db),
            "row_group_size": s.row_group_size,
            "batch_size": s.batch_size,
            "video_cache_frames": s.video_cache_frames,
        },
        "Configuration",
    )


# -- observability ----------------------------------------------------------
@obs_app.command("runs")
def obs_runs(
    dataset_id: str = typer.Argument(...),
    limit: int = typer.Option(20, help="How many runs to show"),
):
    """Recent pipeline runs, newest first.

    The question the engine could not answer before: runs were JSON files on
    disk, not rows, so there was no time series to look at.
    """
    rows = _catalog().run_history(dataset_id, limit)
    if not rows:
        console.print("[yellow]no runs recorded for this dataset[/yellow]")
        raise typer.Exit(0)
    _table(
        [{"run": r["run_id"][-12:], "started": (r["started_at"] or "")[:19],
          "status": r["status"], "seconds": round(r["duration_s"] or 0.0, 2),
          "config": (r["config_hash"] or "")[:8]} for r in rows],
        f"Pipeline runs -- {dataset_id}",
    )


@obs_app.command("stages")
def obs_stages(
    dataset_id: str = typer.Argument(...),
    stage: Optional[str] = typer.Option(None, help="Restrict to one stage"),
    limit: int = typer.Option(30),
):
    """Stage timings across runs -- has this stage gotten slower?"""
    rows = _catalog().stage_history(dataset_id, stage, limit)
    if not rows:
        console.print("[yellow]no stage records[/yellow]")
        raise typer.Exit(0)
    _table(
        [{"started": (r["run_started_at"] or "")[:19], "stage": r["stage"],
          "status": r["status"], "seconds": round(r["duration_s"] or 0.0, 3),
          "frames": r["records_out"], "episodes": r["episodes"],
          "rss_mb": round((r["peak_rss_bytes"] or 0) / 1e6, 1)} for r in rows],
        f"Stage history -- {dataset_id}" + (f" / {stage}" if stage else ""),
    )


@obs_app.command("quality")
def obs_quality(
    dataset_id: str = typer.Argument(...),
    version: Optional[str] = typer.Option(None),
    limit: int = typer.Option(20),
):
    """QC executions over time -- did quality change, and when?

    Findings themselves are replaced on re-validation; these execution headers
    accumulate, so the trend survives a re-run.
    """
    rows = _catalog().qc_execution_history(dataset_id, version, limit)
    if not rows:
        console.print("[yellow]no QC executions recorded[/yellow]")
        raise typer.Exit(0)
    _table(
        [{"started": (r["started_at"] or "")[:19], "version": r["version"],
          "checks": r["checks_run"], "findings": r["findings"],
          "passed": r["episodes_passed"], "warned": r["episodes_warned"],
          "rejected": r["episodes_rejected"], "status": r["dataset_status"]}
         for r in rows],
        f"QC executions -- {dataset_id}",
    )


@obs_app.command("metrics")
def obs_metrics(
    show: bool = typer.Option(False, "--show", help="Print the exposition"),
    max_age_minutes: float = typer.Option(
        60.0, help="Age beyond which the export counts as stale"
    ),
):
    """Inspect the exported Prometheus textfile.

    Freshness is the file's mtime, which is exactly the property a Pushgateway
    cannot provide: with no `up` metric, "the job never ran" and "the job ran
    long ago" are indistinguishable there. Here a missing file is stale by
    definition.
    """
    from .observability.exporters import stale_after

    path = get_settings().paths.data_root / "observability" / "metrics" / "rde.prom"
    if not path.exists():
        console.print(f"[yellow]no export at {path}[/yellow]")
        raise typer.Exit(1)
    text = path.read_text()
    series = [l for l in text.splitlines() if l and not l.startswith("#")]
    stale = stale_after(path, max_age_minutes * 60)
    _print({
        "path": str(path),
        "series": len(series),
        "metrics": sum(1 for l in text.splitlines() if l.startswith("# TYPE")),
        "stale": stale,
    })
    if stale:
        console.print(
            f"[red]STALE[/red]: no successful export in the last "
            f"{max_age_minutes:g} minutes"
        )
    if show:
        console.print(text)


@obs_app.command("health")
def obs_health(
    dataset_id: str = typer.Argument(...),
    version: str = typer.Argument("raw/v1"),
    limit: int = typer.Option(15, help="Worst-N episodes to show"),
):
    """Per-episode health, worst first, from the analytical event rows.

    The episode-grained facts deliberately live here rather than in metric
    labels: at the planning scale that choice is the difference between ~167 TB
    of resident memory and 6 GB of Parquet (docs/14 §3).
    """
    import duckdb

    root = get_settings().paths.data_root / "observability" / "events" / "episode_health"
    files = sorted(root.glob("*.parquet")) if root.exists() else []
    if not files:
        console.print("[yellow]no episode health rows -- run the pipeline first[/yellow]")
        raise typer.Exit(1)
    # Rows accumulate one file per run, so the same episode appears once per
    # validation. Scoping to the newest qc_execution_id is what makes this a
    # "current health" view rather than a pile of history -- the history is
    # still there, and `obs quality` is the view that uses it.
    rows = duckdb.connect().execute(
        f"""WITH latest AS (
                SELECT qc_execution_id
                FROM read_parquet('{root}/*.parquet')
                WHERE dataset_id = ? AND version = ?
                ORDER BY checked_at DESC
                LIMIT 1
            )
            SELECT episode_id, quality_status, frame_count,
                   round(duration_s,1) AS duration_s,
                   round(temporal_quality,2) AS temporal,
                   round(numeric_quality,2) AS numeric,
                   round(alignment_quality,2) AS alignment, qc_bitmask
            FROM read_parquet('{root}/*.parquet')
            WHERE dataset_id = ? AND version = ?
              AND qc_execution_id = (SELECT qc_execution_id FROM latest)
            ORDER BY (temporal_quality + numeric_quality + alignment_quality) ASC,
                     episode_id ASC
            LIMIT {int(limit)}""",
        [dataset_id, version, dataset_id, version],
    ).fetchall()
    if not rows:
        console.print(f"[yellow]no rows for {dataset_id}@{version}[/yellow]")
        raise typer.Exit(1)
    cols = ["episode_id", "quality_status", "frame_count", "duration_s",
            "temporal", "numeric", "alignment", "qc_bitmask"]
    _table([dict(zip(cols, r)) for r in rows],
           f"Episode health -- {dataset_id}@{version} (worst first)")


@obs_app.command("drift")
def obs_drift(
    dataset_id: str = typer.Argument(...),
    baseline: str = typer.Argument(...),
    candidate: str = typer.Argument(...),
    statistic: str = typer.Option("psi,mean_shift", help="Comma-separated"),
):
    """Compare two versions' episode-feature distributions.

    The unit of analysis is the episode, never the frame: at 30-50 fps adjacent
    frames are not independent observations, and an IID test over tens of
    thousands of them reports significance for physically meaningless
    differences (docs/13 §5). Effective sample size is printed beside every
    statistic, because a large value computed on a handful of effective episodes
    is noise wearing a p-value.
    """
    import json as _json

    import duckdb
    import numpy as np

    from .observability.drift import compare

    root = get_settings().paths.data_root / "observability" / "events" / "episode_health"
    if not root.exists():
        console.print("[yellow]no episode health rows[/yellow]")
        raise typer.Exit(1)

    def load(version: str) -> dict:
        rows = duckdb.connect().execute(
            f"""WITH latest AS (
                    SELECT qc_execution_id FROM read_parquet('{root}/*.parquet')
                    WHERE dataset_id = ? AND version = ?
                    ORDER BY checked_at DESC LIMIT 1
                )
                SELECT drift_features_json FROM read_parquet('{root}/*.parquet')
                WHERE dataset_id = ? AND version = ?
                  AND qc_execution_id = (SELECT qc_execution_id FROM latest)""",
            [dataset_id, version, dataset_id, version],
        ).fetchall()
        feats = [_json.loads(r[0]) for r in rows if r[0]]
        if not feats:
            return {}
        keys = [k for k in sorted(feats[0]) if all(k in f for f in feats)]
        return {k: np.array([f[k] for f in feats], dtype=float) for k in keys}

    b, c = load(baseline), load(candidate)
    if not b or not c:
        console.print(
            f"[yellow]missing features for {baseline!r} or {candidate!r}[/yellow]"
        )
        raise typer.Exit(1)

    results = compare(b, c, statistics=tuple(s.strip() for s in statistic.split(",")))
    _table(
        [{"feature": r.feature, "stat": r.statistic, "value": round(r.value, 4),
          "threshold": round(r.threshold, 4), "n_eff": round(r.effective_sample_size, 1),
          "verdict": "EXCEEDED" if r.exceeded
                     else ("underpowered" if r.underpowered else "ok")}
         for r in sorted(results, key=lambda r: -r.value)],
        f"Drift -- {baseline} -> {candidate}",
    )
    exceeded = [r for r in results if r.exceeded]
    console.print(
        f"[red]{len(exceeded)} feature(s) outside contract[/red]" if exceeded
        else "[green]within contract[/green]"
    )


@obs_app.command("contract")
def obs_contract(
    dataset_id: str = typer.Argument(...),
    baseline: str = typer.Argument(..., help="Version to infer the contract from"),
    candidate: Optional[str] = typer.Option(
        None, help="Version to check against it; omit to only infer"
    ),
    out: Optional[Path] = typer.Option(None, help="Where to write the contract JSON"),
):
    """Infer a data health contract, and optionally check a version against it.

    Contracts are the third policy layer. A *check* reports what it saw, a
    *gate* decides what that means for one episode, and a *contract* decides
    whether a whole version may be released to consumers. Collapsing the three
    is how a system ends up unable to ship a dataset that is 99% fine, or
    unable to stop one that is subtly wrong.

    Thresholds are calibrated against the baseline's own null distribution
    rather than borrowed from convention -- PSI's conventional 0.25 fires on
    62.5% of comparisons of a distribution against itself at episode scale
    (docs/15 section 7).

    Contracts are per stage. One inferred from raw/vN will correctly reject
    clean/vN if a transform resized images, because that is a real schema
    change -- an intended one, but the contract cannot know that. Infer the
    contract from the stage you intend to release.
    """
    import numpy as np

    from .observability import DataContract, render_report
    from .observability.drift import compare
    from .observability.health import SHARED_KEY
    from .qc import QCEngine

    settings = get_settings()
    catalog = _catalog()
    # collect_features=True: contract inference needs the baseline's episode
    # feature distribution, but is not itself an observed pipeline run.
    engine = QCEngine(settings=settings, catalog=catalog, collect_features=True)

    def _arrays(feats: dict) -> dict:
        keys = sorted({k for f in feats.values() for k in f})
        return {
            k: np.array([f[k] for f in feats.values() if k in f], dtype=float)
            for k in keys if sum(k in f for f in feats.values()) == len(feats)
        }

    base_res = engine.run(dataset_id, baseline, persist=False)
    base_feats = _arrays(base_res.shared.get(SHARED_KEY, {}))
    manifest = catalog.get_manifest(dataset_id, baseline)
    contract = DataContract.infer(
        manifest, base_res.episode_status, base_feats,
        name=f"{dataset_id}@{baseline}",
    )
    _print({
        "contract": contract.name,
        "baseline": contract.baseline_version,
        "required_modalities": contract.required_modalities,
        "fps": contract.fps,
        "max_rejection_ratio": contract.max_rejection_ratio,
        "calibrated_features": len(contract.drift_thresholds),
    }, "Inferred contract")

    if out:
        console.print(f"[green]written:[/green] {contract.save(out)}")

    if not candidate:
        raise typer.Exit(0)

    cand_res = engine.run(dataset_id, candidate, persist=False)
    drift = compare(
        base_feats, _arrays(cand_res.shared.get(SHARED_KEY, {})),
        statistics=("psi",), thresholds=contract.drift_thresholds,
    )
    report = contract.evaluate(
        catalog.get_manifest(dataset_id, candidate),
        cand_res.episode_status,
        alignment_offsets=cand_res.shared.get("alignment_offsets"),
        drift_results=drift,
    )
    console.print(escape(render_report(report)))
    raise typer.Exit(0 if report.released else 2)


@obs_app.command("video")
def obs_video(
    dataset_id: str = typer.Argument(...),
    version: str = typer.Argument("raw/v1"),
    tier: int = typer.Option(0, help="0 = headers only, 1 = sparse decode"),
    limit: int = typer.Option(10, help="Episode-camera references to sample at tier 1"),
):
    """Video health without decoding everything.

    SO-100 is 470 MB on disk and 36.2 GB decoded -- 77x. At the planning scale
    of 3-10 PB/year, a strategy that decodes everything costs more than the
    training it supports.

    Tier 0 reads container headers only (~2 ms/file) and already catches a
    resolution change, a codec swap, a truncated file and a frame-count
    disagreement -- the failures a changed recording rig produces, which hit
    whole cohorts at once. Tier 1 decodes a handful of frames per clip
    (~90 ms) for brightness, sharpness, black frames and frozen streams, at
    one frame of peak memory.
    """
    import json as _json

    from .observability.video_health import cohort_summary, tier0, tier1
    from .storage import BlobStore, ParquetTableStore

    settings = get_settings()
    catalog = _catalog()
    version_dir = settings.paths.version_dir(dataset_id, version)
    tables = ParquetTableStore(version_dir)
    if not tables.exists("episodes"):
        console.print(f"[yellow]no episode table for {dataset_id}@{version}[/yellow]")
        raise typer.Exit(1)

    blobs = BlobStore(settings.paths.blobs)
    et = tables.read_table("episodes", columns=["episode_id", "blob_ids"]).to_pydict()
    refs = [
        (eid, modality, blob_id)
        for eid, raw in zip(et["episode_id"], et["blob_ids"])
        for modality, blob_id in _json.loads(raw or "{}").items()
    ]
    if not refs:
        console.print("[yellow]this version has no external (video) modalities[/yellow]")
        raise typer.Exit(0)

    if tier == 0:
        distinct = {blob_id: modality for _, modality, blob_id in refs}
        reports = [tier0(blobs.get(b), b, m) for b, m in distinct.items()]
        _print(cohort_summary(reports), f"Tier 0 -- {dataset_id}@{version}")
        bad = [r for r in reports if not r.ok]
        if bad:
            _table([{"blob": r.blob_id[:12], "modality": r.modality,
                     "error": r.error} for r in bad], "Unreadable containers")
        raise typer.Exit(1 if bad else 0)

    episodes = {r["episode_id"]: r for r in catalog.get_episodes(dataset_id, version)}
    rows = []
    for eid, modality, blob_id in refs[:limit]:
        attrs = _json.loads(episodes.get(eid, {}).get("attributes") or "{}")
        win = (attrs.get("video_windows") or {}).get(modality)
        window = (
            (float(win["from_timestamp"]), float(win["to_timestamp"])) if win else None
        )
        r = tier1(blobs.get(blob_id), blob_id, modality, window=window)
        rows.append({
            "episode": eid, "modality": r.modality, "sampled": r.frames_sampled,
            "brightness": round(r.brightness_mean, 1),
            "contrast": round(r.contrast_mean, 1),
            "sharpness": round(r.sharpness_mean, 1),
            "interframe": round(r.interframe_difference, 1),
            "black": round(r.black_frame_ratio, 2),
            "frozen": r.frozen,
            "dupes": r.duplicate_samples,
        })
    _table(rows, f"Tier 1 -- {dataset_id}@{version} (first {len(rows)} references)")


def main():
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()


