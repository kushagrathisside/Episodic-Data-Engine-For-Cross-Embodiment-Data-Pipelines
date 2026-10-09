"""Data health contracts: the third policy layer.

The engine already separates two things that are usually conflated:

* a **check** reports what it saw (``Severity`` on the check, ``CheckStatus`` on
  the evaluation);
* a **gate** decides what that means for one episode (:class:`QCGate`).

A contract is the third: it decides whether a *dataset version* may be
**released** to consumers.  The three are genuinely different questions with
different owners, and collapsing them is how a system ends up unable to ship a
dataset that is 99% fine or unable to stop one that is subtly wrong.

    check   →  "episode 17 has a 2.1 s timestamp gap"        (an observation)
    gate    →  "episode 17 is rejected from clean/v1"        (per episode)
    contract→  "training/v3 may not be released: action
                magnitude drifted 4.1σ from the baseline"    (per version)

Five clause families:

``schema``        fields, dtypes, dimensions, required modalities
``temporal``      declared sampling rate, jitter, alignment tolerance
``quality``       missingness, NaN rate, rejection ratio
``distribution``  episode-feature drift against a named baseline
``release``       which of the above block, and which merely warn

**Contracts are inferred, not hand-written.**  A contract typed from
imagination encodes what someone assumed the data looked like.
:meth:`DataContract.infer` derives one from a version that was actually
inspected, which is the only kind whose numbers mean anything.  It is then
reviewed and edited by a human -- the inference gives you a defensible starting
point, not the final word.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..schemas.core import DatasetManifest, QualityStatus

#: Clause outcome.  ``SKIPPED`` matters: a distribution clause with no baseline
#: has not passed, it has not been evaluated, and reporting the two the same way
#: is how a contract quietly stops protecting anything.
PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"


@dataclass
class Clause:
    """One contractual assertion and its outcome."""

    family: str
    name: str
    outcome: str
    observed: str = ""
    expected: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome in (PASS, SKIP, WARN)


@dataclass
class ContractReport:
    """The result of evaluating one contract against one dataset version."""

    dataset_id: str
    version: str
    contract_name: str
    evaluated_at: str
    clauses: list[Clause] = field(default_factory=list)
    #: Clause families whose failure blocks release, from the release policy.
    blocking_families: tuple[str, ...] = ()

    @property
    def failures(self) -> list[Clause]:
        return [c for c in self.clauses if c.outcome == FAIL]

    @property
    def warnings(self) -> list[Clause]:
        return [c for c in self.clauses if c.outcome == WARN]

    @property
    def blocking(self) -> list[Clause]:
        """Failures in families the release policy treats as blocking."""
        return [c for c in self.failures if c.family in self.blocking_families]

    @property
    def released(self) -> bool:
        """Whether this version may go to consumers."""
        return not self.blocking

    def summary(self) -> dict[str, Any]:
        """Compact machine-readable verdict."""
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "contract": self.contract_name,
            "released": self.released,
            "clauses": len(self.clauses),
            "failures": len(self.failures),
            "warnings": len(self.warnings),
            "blocking": [f"{c.family}.{c.name}" for c in self.blocking],
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class DataContract:
    """What a dataset version must be true of before anyone may train on it."""

    name: str
    dataset_id: str
    #: The version this contract was inferred from.  Recorded because a
    #: distribution clause is meaningless without saying what it is relative to.
    baseline_version: str = ""
    created_at: str = ""

    # -- schema -----------------------------------------------------------
    required_modalities: list[str] = field(default_factory=list)
    modality_dtypes: dict[str, str] = field(default_factory=dict)
    modality_shapes: dict[str, list[int]] = field(default_factory=dict)

    # -- temporal ---------------------------------------------------------
    fps: float | None = None
    #: Fractional tolerance on the declared sampling rate.
    fps_tolerance: float = 0.02
    #: Alignment tolerance, expressed in frame periods rather than seconds so
    #: the same contract is meaningful for a 30 fps and a 50 fps dataset.
    alignment_tolerance_frames: float = 0.5

    # -- quality ----------------------------------------------------------
    max_missing_ratio: float = 0.01
    max_rejection_ratio: float = 0.10
    min_episodes: int = 1

    # -- distribution -----------------------------------------------------
    #: feature -> threshold, normally produced by ``drift.calibrate_threshold``
    #: against the baseline's own null rather than borrowed from convention.
    drift_thresholds: dict[str, float] = field(default_factory=dict)
    drift_statistic: str = "psi"

    # -- release policy ---------------------------------------------------
    #: Families whose failures block a release.  Distribution is deliberately
    #: blocking by default: a silently rescaled controller produces data that
    #: passes every structural check and should not reach a training run.
    blocking_families: tuple[str, ...] = ("schema", "temporal", "quality", "distribution")

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form."""
        d = asdict(self)
        d["blocking_families"] = list(self.blocking_families)
        return d

    def save(self, path: Path) -> Path:
        """Write the contract as JSON.  Contracts are reviewed artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "DataContract":
        """Read a contract back from JSON."""
        d = json.loads(Path(path).read_text())
        d["blocking_families"] = tuple(d.get("blocking_families", ()))
        return cls(**d)

    # ----------------------------------------------------------- inference
    @classmethod
    def infer(
        cls,
        manifest: DatasetManifest,
        episode_status: dict[int, Any] | None = None,
        baseline_features: dict[str, Any] | None = None,
        name: str = "",
        calibrate: bool = True,
    ) -> "DataContract":
        """Derive a contract from a version that was actually inspected.

        Distribution thresholds are measured from the baseline's own null
        distribution rather than borrowed.  That matters more than it sounds:
        PSI's conventional 0.25 threshold, applied to 50 episodes, fires on
        62.5% of comparisons of a distribution *against itself* (docs/15 §7).
        A contract built on it would block almost every release and teach
        everyone to ignore it.
        """
        from .drift import calibrate_threshold

        statuses = list((episode_status or {}).values())
        rejected = sum(1 for s in statuses if s is QualityStatus.REJECTED)
        observed_rejection = rejected / len(statuses) if statuses else 0.0

        thresholds: dict[str, float] = {}
        if baseline_features and calibrate:
            import numpy as np

            for feature, values in baseline_features.items():
                arr = np.asarray(values, dtype=float)
                if arr.size >= 20:
                    thresholds[feature] = round(
                        float(calibrate_threshold(arr, "psi")), 6
                    )

        return cls(
            name=name or f"{manifest.dataset_id}-contract",
            dataset_id=manifest.dataset_id,
            baseline_version=manifest.version,
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            required_modalities=sorted(m.name for m in manifest.modalities),
            modality_dtypes={m.name: str(m.dtype) for m in manifest.modalities},
            modality_shapes={
                m.name: list(m.shape) for m in manifest.modalities if m.shape
            },
            fps=manifest.fps,
            # Allow headroom above what the baseline actually showed, so a
            # contract inferred from a good day does not fail on a normal one.
            max_rejection_ratio=round(max(0.10, observed_rejection * 2.0), 4),
            min_episodes=max(1, len(statuses) // 2),
            drift_thresholds=thresholds,
        )

    # ---------------------------------------------------------- evaluation
    def evaluate(
        self,
        manifest: DatasetManifest,
        episode_status: dict[int, Any] | None = None,
        missing_ratios: dict[str, float] | None = None,
        alignment_offsets: Sequence[tuple[str, float]] | None = None,
        observed_fps: float | None = None,
        drift_results: Sequence[Any] | None = None,
    ) -> ContractReport:
        """Check one version against every clause and return a release verdict."""
        report = ContractReport(
            dataset_id=manifest.dataset_id,
            version=manifest.version,
            contract_name=self.name,
            evaluated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            blocking_families=self.blocking_families,
        )
        add = report.clauses.append

        # -- schema --------------------------------------------------------
        present = {m.name for m in manifest.modalities}
        for required in self.required_modalities:
            add(Clause(
                "schema", f"modality_present[{required}]",
                PASS if required in present else FAIL,
                observed="present" if required in present else "absent",
                expected="present",
            ))
        for name, expected_dtype in self.modality_dtypes.items():
            found = next((m for m in manifest.modalities if m.name == name), None)
            if found is None:
                continue  # already reported by the presence clause
            actual = str(found.dtype)
            add(Clause(
                "schema", f"dtype[{name}]",
                PASS if actual == expected_dtype else FAIL,
                observed=actual, expected=expected_dtype,
            ))
        for name, expected_shape in self.modality_shapes.items():
            found = next((m for m in manifest.modalities if m.name == name), None)
            if found is None or not found.shape:
                continue
            actual_shape = list(found.shape)
            add(Clause(
                "schema", f"shape[{name}]",
                PASS if actual_shape == list(expected_shape) else FAIL,
                observed=str(actual_shape), expected=str(list(expected_shape)),
                detail="a changed dimension silently changes what the model consumes",
            ))

        # -- temporal ------------------------------------------------------
        if self.fps:
            declared = manifest.fps or 0.0
            drift_frac = abs(declared - self.fps) / self.fps
            add(Clause(
                "temporal", "declared_fps",
                PASS if drift_frac <= self.fps_tolerance else FAIL,
                observed=f"{declared:g}", expected=f"{self.fps:g} ±{self.fps_tolerance:.0%}",
            ))
            if observed_fps:
                obs_frac = abs(observed_fps - self.fps) / self.fps
                add(Clause(
                    "temporal", "observed_fps",
                    PASS if obs_frac <= self.fps_tolerance else FAIL,
                    observed=f"{observed_fps:.4g}",
                    expected=f"{self.fps:g} ±{self.fps_tolerance:.0%}",
                    detail="a stream running uniformly at the wrong rate is "
                           "self-consistent and still wrong",
                ))
        if alignment_offsets is not None:
            period = 1.0 / max(self.fps or manifest.fps or 30.0, 1e-9)
            limit = self.alignment_tolerance_frames * period
            worst = max((abs(o) for _, o in alignment_offsets), default=0.0)
            add(Clause(
                "temporal", "cross_modal_alignment",
                PASS if worst <= limit else FAIL,
                observed=f"{worst * 1000:.3f} ms",
                expected=f"<= {limit * 1000:.1f} ms ({self.alignment_tolerance_frames} frame)",
                detail="the pillar the canonical five do not have",
            ))

        # -- quality -------------------------------------------------------
        statuses = list((episode_status or {}).values())
        if statuses:
            rejected = sum(1 for s in statuses if s is QualityStatus.REJECTED)
            ratio = rejected / len(statuses)
            add(Clause(
                "quality", "rejection_ratio",
                PASS if ratio <= self.max_rejection_ratio else FAIL,
                observed=f"{ratio:.2%}", expected=f"<= {self.max_rejection_ratio:.2%}",
            ))
            surviving = len(statuses) - rejected
            add(Clause(
                "quality", "min_episodes",
                PASS if surviving >= self.min_episodes else FAIL,
                observed=str(surviving), expected=f">= {self.min_episodes}",
            ))
        for name, ratio in (missing_ratios or {}).items():
            add(Clause(
                "quality", f"missing_ratio[{name}]",
                PASS if ratio <= self.max_missing_ratio else FAIL,
                observed=f"{ratio:.3%}", expected=f"<= {self.max_missing_ratio:.3%}",
            ))

        # -- distribution --------------------------------------------------
        if not self.drift_thresholds:
            add(Clause(
                "distribution", "baseline",
                SKIP, observed="no calibrated thresholds",
                expected="a baseline version",
                detail="not evaluated -- which is not the same as passing",
            ))
        else:
            evaluated = 0
            for r in drift_results or ():
                if r.statistic != self.drift_statistic:
                    continue
                limit = self.drift_thresholds.get(r.feature)
                if limit is None:
                    continue
                evaluated += 1
                if r.underpowered:
                    add(Clause(
                        "distribution", f"drift[{r.feature}]",
                        WARN, observed=f"{r.value:.4f}", expected=f"<= {limit:.4f}",
                        detail=f"underpowered: n_eff={r.effective_sample_size:.1f}",
                    ))
                    continue
                add(Clause(
                    "distribution", f"drift[{r.feature}]",
                    PASS if r.value <= limit else FAIL,
                    observed=f"{r.value:.4f}", expected=f"<= {limit:.4f}",
                    detail=f"n_eff={r.effective_sample_size:.1f}",
                ))
            if evaluated == 0:
                add(Clause(
                    "distribution", "coverage",
                    SKIP, observed="0 features compared",
                    expected=f"{len(self.drift_thresholds)} features",
                    detail="no drift results were supplied for the contracted features",
                ))
        return report


def render_report(report: ContractReport) -> str:
    """Human-readable contract verdict, failures first."""
    lines = [
        f"contract {report.contract_name} vs {report.dataset_id}@{report.version}",
        f"  verdict : {'RELEASE' if report.released else 'BLOCKED'}",
        f"  clauses : {len(report.clauses)} "
        f"({len(report.failures)} failed, {len(report.warnings)} warned)",
    ]
    ordered = (
        report.failures
        + report.warnings
        + [c for c in report.clauses if c.outcome == SKIP]
    )
    for c in ordered[:20]:
        lines.append(
            f"    [{c.outcome.upper():5s}] {c.family}.{c.name}  "
            f"observed={c.observed} expected={c.expected}"
        )
    return "\n".join(lines)
