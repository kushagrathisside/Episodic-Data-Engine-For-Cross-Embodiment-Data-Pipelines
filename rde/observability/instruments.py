"""The declared metric catalog: every series this engine may emit.

Declared in one place, on purpose.  Metrics scattered across modules become
metrics nobody can enumerate, and a label policy you cannot enumerate is a label
policy you cannot audit.  Every instrument here has a documented label set whose
cardinality was computed in docs/14 §3 rather than guessed.

Naming follows Prometheus convention -- ``rde_<subsystem>_<thing>_<unit>``,
counters suffixed ``_total``, durations in seconds, sizes in bytes -- so the
output drops into any Prometheus-compatible store without translation.

Three families:

``pipeline_*``   Is the system behaving normally?  Throughput, latency, failures.
``qc_*``         Is validation behaving normally?  Findings, statuses, check cost.
``data_*``       Is the *data* healthy?  Coverage, alignment, sensor health, drift.

The third family is the one general-purpose observability tooling does not have.
A pipeline can succeed perfectly and emit a ruined dataset -- our
``inject_video_misalignment`` fault shifts video by 5 s and every structural,
schema, volume and null check still passes.  ``data_*`` exists to make that
visible (docs/13 §7).
"""

from __future__ import annotations

from .metrics import DURATION_BUCKETS, MetricRegistry, SIZE_BUCKETS

#: Fraction-valued buckets for ratios in [0, 1] -- coverage, missingness.
RATIO_BUCKETS: tuple[float, ...] = (
    0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0,
)

#: Alignment offsets in seconds.  Resolution matters at the low end: at 30 fps
#: one frame period is 33 ms, so the buckets below it are where a genuinely
#: sub-frame offset is distinguished from a real desynchronisation.
OFFSET_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.033, 0.067, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0,
)


def build_registry(registry: MetricRegistry | None = None) -> MetricRegistry:
    """Declare every instrument.  Idempotent -- safe to call repeatedly."""
    r = registry or MetricRegistry()

    # ---------------------------------------------------------------- pipeline
    # "Is the system behaving normally?"  Labels: stage, outcome, error_class,
    # dataset_id.  Cardinality ceiling 8 x 2 x 13 x |datasets|.
    r.counter("rde_pipeline_runs_total",
              "Pipeline executions, by terminal outcome.")
    r.counter("rde_pipeline_stage_failures_total",
              "Stage failures, by stage and normalised exception class.")
    r.histogram("rde_pipeline_stage_duration_seconds",
                "Wall-clock duration of one pipeline stage.", DURATION_BUCKETS)
    r.histogram("rde_pipeline_run_duration_seconds",
                "Wall-clock duration of a whole pipeline run.", DURATION_BUCKETS)
    r.counter("rde_pipeline_frames_processed_total",
              "Frames passing through a stage.")
    r.counter("rde_pipeline_episodes_processed_total",
              "Episodes passing through a stage.")
    r.counter("rde_pipeline_bytes_read_total", "Bytes read by a stage.")
    r.counter("rde_pipeline_bytes_written_total", "Bytes written by a stage.")
    r.gauge("rde_pipeline_stage_peak_rss_bytes",
            "Peak resident set size observed during a stage.")

    # Batch-idiomatic gauges.  A batch job's process exits, so its counters
    # restart at zero on the next run; Prometheus detects that as a counter
    # reset, but `increase()` over a window is only correct if every run was
    # scraped, which for a job that finishes in 7 seconds it will not be.
    # The reliable signals for a batch job are therefore point-in-time facts
    # about the *last* run, which is what these carry.  The counters above stay
    # -- they are correct within a run and they are what the event rows are
    # aggregated from -- but alerts should be built on these.
    r.gauge("rde_pipeline_last_success_timestamp_seconds",
            "Unix time of the last successful run. The freshness signal: "
            "alert on time() - this, not on a rate.")
    r.gauge("rde_pipeline_last_run_timestamp_seconds",
            "Unix time of the last run, successful or not.")
    r.gauge("rde_pipeline_last_run_duration_seconds",
            "Duration of the last run of a stage. Gauge, not histogram, "
            "because one run contributes exactly one observation.")
    r.gauge("rde_pipeline_last_run_status",
            "1 if the last run succeeded, 0 if it failed.")
    r.gauge("rde_pipeline_last_run_frames",
            "Frames processed by a stage in the last run. The volume pillar: "
            "a stage that consumed 55,000 and emitted 12 raised no exception.")
    r.gauge("rde_pipeline_last_run_episodes",
            "Episodes processed by a stage in the last run.")

    # -------------------------------------------------------------------- qc
    # Labels: check_name(24), severity(4), status(4), scope(3), dataset_id.
    r.counter("rde_qc_findings_total",
              "QC findings, by check, severity and status.")
    r.counter("rde_qc_episode_status_total",
              "Episodes reaching each quality status after gating.")
    r.histogram("rde_qc_check_duration_seconds",
                "Time spent inside one check across the whole episode pass.",
                DURATION_BUCKETS)
    r.counter("rde_qc_checks_run_total", "Check executions.")
    r.gauge("rde_qc_gate_rejection_ratio",
            "Fraction of episodes rejected by the gate in the last run.")

    # ------------------------------------------------------------------ data
    # Data health.  The family that distinguishes "the pipeline ran" from
    # "the data is usable".
    r.gauge("rde_data_modality_coverage_ratio",
            "Fraction of episodes in which a modality is present.")
    r.gauge("rde_data_missing_ratio",
            "Fraction of expected values that are absent or null.")
    r.gauge("rde_data_sampling_interval_seconds",
            "Median inter-frame interval observed for a modality.")
    r.histogram("rde_data_sampling_jitter_seconds",
                "Deviation of inter-frame interval from its declared period.",
                OFFSET_BUCKETS)
    r.histogram("rde_data_timestamp_offset_seconds",
                "Cross-modal timestamp offset. The alignment pillar.",
                OFFSET_BUCKETS)
    r.counter("rde_data_decode_failures_total",
              "Video frames that failed to decode.")
    r.counter("rde_data_nonfinite_values_total",
              "NaN or Inf values observed, by modality.")
    r.histogram("rde_data_episode_duration_seconds",
                "Episode duration distribution.", DURATION_BUCKETS)
    r.histogram("rde_data_episode_frame_count",
                "Episode frame-count distribution.",
                (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000))

    # Sensor health inferred from the data itself, because hardware telemetry
    # is not available to a dataset engine.  These are *data-derived health
    # indicators*, never claims of direct hardware diagnosis.
    r.counter("rde_sensor_stuck_intervals_total",
              "Intervals where a channel's value did not change.")
    r.counter("rde_sensor_saturation_events_total",
              "Observations at or beyond a channel's physical rail.")
    r.gauge("rde_sensor_tail_ratio",
            "max / p99.5 for a channel. >1 is normal for impulsive modalities; "
            "a sudden shift is the signal, not the level.")
    r.counter("rde_sensor_frame_dropout_total",
              "Missing frames inferred from index or timestamp discontinuity.")

    # Drift.  ``feature`` is bounded by the extractor, which is code.
    r.gauge("rde_data_distribution_drift",
            "Drift statistic for one episode-level feature against a baseline.")
    r.gauge("rde_data_drift_effective_sample_size",
            "Episodes contributing to a drift comparison. Small values mean the "
            "statistic is not trustworthy, and must be published alongside it.")

    # ------------------------------------------------------- self-observation
    # A metric layer that cannot report its own cost is the one that surprises
    # you.  This is the metric that catches a cardinality regression in review.
    r.gauge("rde_observability_series_count",
            "Total active series held by this registry.")
    r.counter("rde_observability_export_failures_total",
              "Telemetry exports that failed. Never fails the pipeline.")

    return r


#: Process-wide registry.  Modules take instruments from here at call time
#: rather than at import time, so a test can swap the registry out.
REGISTRY: MetricRegistry = build_registry()
