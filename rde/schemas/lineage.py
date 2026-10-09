"""Lineage and versioning records.

Two questions must always be answerable:

1. *Which raw episodes contributed to this training dataset?*
2. *What transformations and QC checks were applied before this dataset reached
   the model?*

Answering (1) needs episode-level provenance, not just dataset-level edges -- a
clean version drops individual episodes, so "clean/v1 came from raw/v1" is not
enough.  Hence :class:`TransformRun` carries the surviving episode ids, and the
catalog stores an episode-level contribution table.

Answering (2) needs the edges to be ordered and parameterised, which is what
``parameters`` plus ``software_version`` and ``code_fingerprint`` give.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VersionStage(str, Enum):
    """Canonical pipeline stages.  Version strings are ``<stage>/v<N>``."""

    RAW = "raw"
    VALIDATED = "validated"
    CLEAN = "clean"
    TRANSFORMED = "transformed"
    TRAINING = "training"
    REJECTED = "rejected"

    @classmethod
    def parse(cls, version: str) -> tuple["VersionStage", int]:
        """Split ``'training/v2'`` into ``(VersionStage.TRAINING, 2)``."""
        if "/" not in version:
            raise ValueError(f"version must be '<stage>/v<N>', got {version!r}")
        stage_s, num_s = version.split("/", 1)
        if not num_s.startswith("v") or not num_s[1:].isdigit():
            raise ValueError(f"version must be '<stage>/v<N>', got {version!r}")
        return cls(stage_s), int(num_s[1:])


class TransformRun(BaseModel):
    """One recorded edge in the lineage graph.

    Emitted by every stage that produces a new version -- ingestion, QC gating,
    each transformation operator, and the dataset builder.
    """

    run_id: str
    dataset_id: str
    operation: str = Field(..., description="Operator name, e.g. 'normalize_state'.")
    input_versions: list[str] = Field(default_factory=list)
    output_version: str

    parameters: dict[str, Any] = Field(default_factory=dict)
    started_at: _dt.datetime
    finished_at: _dt.datetime | None = None

    software_version: str = ""
    code_fingerprint: str = Field(
        default="", description="Hash of the operator implementation, for repro."
    )

    #: Episode ids that survived this step, enabling episode-level provenance.
    input_episode_ids: list[int] = Field(default_factory=list)
    output_episode_ids: list[int] = Field(default_factory=list)
    dropped_episode_ids: list[int] = Field(default_factory=list)

    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        """Wall-clock seconds the run took, or None if it never finished."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def fingerprint(self) -> str:
        """Stable hash of (operation, inputs, parameters).

        Two runs with the same fingerprint should produce byte-identical output,
        which is what makes the pipeline reproducible rather than merely logged.
        """
        payload = json.dumps(
            {
                "operation": self.operation,
                "inputs": sorted(self.input_versions),
                "parameters": self.parameters,
                "code": self.code_fingerprint,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class LineageNode(BaseModel):
    """A version as seen by the lineage graph."""

    version_key: str
    dataset_id: str
    version: str
    stage: VersionStage
    created_at: _dt.datetime
    total_episodes: int = 0
    total_frames: int = 0


class LineageGraph(BaseModel):
    """Materialised provenance DAG for one dataset id."""

    dataset_id: str
    nodes: dict[str, LineageNode] = Field(default_factory=dict)
    edges: list[TransformRun] = Field(default_factory=list)

    def ancestors(self, version: str) -> list[str]:
        """All upstream versions of ``version``, nearest first, deduplicated."""
        seen: list[str] = []
        frontier = [version]
        guard = 0
        while frontier and guard < 1000:
            guard += 1
            current = frontier.pop(0)
            for edge in self.edges:
                if edge.output_version == current:
                    for parent in edge.input_versions:
                        if parent not in seen:
                            seen.append(parent)
                            frontier.append(parent)
        return seen

    def path_to_root(self, version: str) -> list[TransformRun]:
        """Ordered chain of runs that produced ``version`` from a raw root."""
        chain: list[TransformRun] = []
        current = version
        guard = 0
        while guard < 1000:
            guard += 1
            producer = next(
                (e for e in self.edges if e.output_version == current), None
            )
            if producer is None:
                break
            chain.append(producer)
            if not producer.input_versions:
                break
            current = producer.input_versions[0]
        return list(reversed(chain))

    def root_episodes(self, version: str) -> list[int]:
        """Raw episode ids that contributed to ``version``.

        Walks back through the run chain intersecting surviving episode sets, so
        the answer reflects what actually made it through the gates.
        """
        chain = self.path_to_root(version)
        if not chain:
            return []
        surviving = set(chain[-1].output_episode_ids)
        for run in reversed(chain):
            if run.input_episode_ids:
                surviving &= set(run.input_episode_ids) | set(run.output_episode_ids)
        return sorted(surviving)
