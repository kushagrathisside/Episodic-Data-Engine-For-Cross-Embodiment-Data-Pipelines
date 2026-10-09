"""Dataset versions and lineage recording.

A *version* is an immutable, named snapshot: ``raw/v1``, ``clean/v1``,
``training/v2``.  Every version except an ingested root is produced by exactly one
recorded :class:`TransformRun`, which is what makes the provenance graph a DAG
rather than a folklore.

What a version does and does not materialise is the central storage decision:

*Materialised per version*: the Parquet frame table, the episode table, the
manifest, and QC artifacts.  These are small -- a few MB for 19 631 frames -- and
having them standalone means a version can be read without replaying its parents.

*Shared across versions*: video blobs, by content address.  A 320 MB camera
container is stored once no matter how many versions reference it.  Copying it
per stage would turn a five-stage pipeline into 2.3 GB of identical bytes, and at
the TB scale this design targets it would be ruinous.

The consequence: promoting ``raw/v1`` to ``training/v1`` costs kilobytes of
metadata, not gigabytes of video.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import inspect as _inspect
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ..catalog import Catalog
from ..config import SOFTWARE_VERSION
from ..schemas.lineage import TransformRun, VersionStage


def code_fingerprint(fn: Callable) -> str:
    """Hash a callable's source so a changed operator yields a changed lineage id.

    Reproducibility claims are only meaningful if the *code* is pinned alongside
    the parameters.  Falls back to the qualified name when source is unavailable
    (C extensions, interactive definitions).
    """
    try:
        src = _inspect.getsource(fn)
    except (OSError, TypeError):
        src = getattr(fn, "__qualname__", repr(fn))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def next_version(catalog: Catalog, dataset_id: str, stage: VersionStage | str) -> str:
    """Return the next unused ``<stage>/v<N>`` for a dataset.

    Not called anywhere in this package's own pipeline (``ingest``/``transform``/
    ``build`` all take an explicit ``output_version`` from the caller); exported
    for callers who want to auto-generate one. It is a non-atomic
    read-then-compute with no locking: two concurrent callers can read the same
    existing versions and compute the same "next" string, and
    :meth:`Catalog.register_version`'s replace-on-conflict upsert will then let
    the second one silently overwrite the first's episodes/frames/manifest with
    no error to either caller. Safe for the sequential, single-process use this
    system is designed for; do not call it from more than one process at a time
    against the same ``(dataset_id, stage)`` without adding your own locking.
    """
    stage_v = VersionStage(stage) if isinstance(stage, str) else stage
    existing = [
        v["version"]
        for v in catalog.list_versions(dataset_id)
        if v["stage"] == stage_v.value
    ]
    n = 0
    for v in existing:
        try:
            _, num = VersionStage.parse(v)
            n = max(n, num)
        except ValueError:
            continue
    return f"{stage_v.value}/v{n + 1}"


class VersionManager:
    """Creates versions and records the runs that produced them."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def resolve(self, dataset_id: str, version: str | None, stage: VersionStage | None = None) -> str:
        """Resolve a version string, or the latest of a stage when omitted."""
        if version:
            return version
        if stage is None:
            raise ValueError("either version or stage must be given")
        versions = [
            v for v in self.catalog.list_versions(dataset_id) if v["stage"] == stage.value
        ]
        if not versions:
            raise LookupError(f"no {stage.value} version exists for {dataset_id}")
        return sorted(versions, key=lambda v: v["created_at"])[-1]["version"]

    @contextmanager
    def run(
        self,
        dataset_id: str,
        operation: str,
        output_version: str,
        input_versions: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        fingerprint: str = "",
        episode_map: dict[int, int] | None = None,
    ) -> Iterator[TransformRun]:
        """Record one lineage edge around a block of work.

        The run is persisted even when the body raises, with the failure captured
        in ``metrics``.  A pipeline that crashed halfway is precisely when you most
        need to know what it was doing.

        ``episode_map`` (output episode id -> input episode id) lets a caller
        override the default identity mapping for a single-input transform that
        renumbers episodes. It is *not* a general multi-source mechanism:
        ``Catalog.record_run`` attributes every mapped episode to
        ``input_versions[0]``, so more than one input version has no correct
        way to record which source each output episode actually came from --
        that combination raises rather than silently attributing every output
        episode to ``input_versions[0]`` by identity, which is what it used to
        do (fabricating plausible-looking but false provenance for any future
        merge-style transform). Recording true multi-source lineage needs a
        richer map (output id -> (source_version, source_episode_id)) and a
        matching change to ``episode_lineage`` writes; nothing in this
        codebase needs that yet.
        """
        run = TransformRun(
            run_id=f"{operation}-{uuid.uuid4().hex[:12]}",
            dataset_id=dataset_id,
            operation=operation,
            input_versions=input_versions or [],
            output_version=output_version,
            parameters=parameters or {},
            started_at=_dt.datetime.now(_dt.timezone.utc),
            software_version=SOFTWARE_VERSION,
            code_fingerprint=fingerprint,
        )
        try:
            yield run
        except Exception as exc:
            run.finished_at = _dt.datetime.now(_dt.timezone.utc)
            run.metrics["error"] = f"{type(exc).__name__}: {exc}"
            self.catalog.record_run(run)
            raise
        else:
            run.finished_at = _dt.datetime.now(_dt.timezone.utc)
            run.metrics.setdefault("duration_s", run.duration_s)
            # Episode-level provenance edges.  An ingested root has no parent
            # version, so it contributes nodes but no edges; every derived
            # version maps each surviving episode back to its source.
            if len(run.input_versions) > 1:
                raise ValueError(
                    f"run.run(operation={operation!r}) has "
                    f"{len(run.input_versions)} input_versions. episode_lineage "
                    "can only attribute mapped episodes to a single source "
                    "version (Catalog.record_run always uses input_versions[0]), "
                    "so recording lineage for a multi-source transform would be "
                    "silently wrong for every episode that didn't come from "
                    "input_versions[0]. Not supported until episode_lineage "
                    "carries a per-episode source version."
                )
            resolved_map = episode_map
            if resolved_map is None and run.output_episode_ids and run.input_versions:
                # Single input version: output episode N provably came from
                # that version's episode N (no renumbering transform exists
                # in this codebase), so identity is a safe default.
                resolved_map = {eid: eid for eid in run.output_episode_ids}
            self.catalog.record_run(run, episode_map=resolved_map)
