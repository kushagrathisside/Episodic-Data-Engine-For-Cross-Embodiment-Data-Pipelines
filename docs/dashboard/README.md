# Dashboard

`data-health-console.html` is the operational dashboard designed in
[docs/15 §8](../15-metrics-catalog.md), built from real telemetry rather than
mocked values. Every figure it renders came out of one pipeline run over the
SO-100 dataset and is reproducible from the repository.

Open it directly in a browser.

## What it shows

| Panel | Source |
|---|---|
| Vitals strip | `rde_pipeline_last_*` gauges + the measured cache speedup |
| Stage timings, volume, peak RSS | `stage_runs` catalog table |
| Per-check duration | `rde_qc_check_duration_seconds` |
| Episode health, 50 cells | `episode_health` Parquet event rows |
| Modality coverage, missingness | `rde_data_modality_coverage_ratio`, `rde_data_missing_ratio` |
| Cross-modal alignment | `rde_data_timestamp_offset_seconds` + the injected-fault measurement |
| Tail ratio, rail hits | `rde_sensor_tail_ratio`, `rde_sensor_saturation_events_total` |
| Cardinality projection | `artifacts/cardinality_experiment.json` |
| QC execution and run history | `qc_executions`, `pipeline_runs` |
| Tiered video health | `rde obs video … --tier 0` / `--tier 1` |
| Release contract verdict | `DataContract.infer` + `evaluate` on a shifted clone |
| Drift calibration evidence | `calibrate_threshold` null distribution + measured ROC |

## Regenerating it

The page embeds a static snapshot, so it is a *demonstration* of the dashboard
design rather than a live view — there is no TSDB behind it, which is stated
plainly in [docs/16 §7](../16-observability-report.md) as a limitation. To
refresh the snapshot, re-run the pipeline and re-export:

```bash
make pipeline
rde obs metrics --show          # the exposition the panels are built from
rde obs health   so100_pickplace raw/v1
rde obs video    so100_pickplace raw/v1 --tier 0
rde obs contract so100_pickplace raw/v1
python examples/08_cardinality_experiment.py
pytest tests/test_observability.py -k drift   # the calibration figures
```

In production the same panels are Grafana queries against VictoriaMetrics
(metrics) and ClickHouse (event rows); the layout and the metric names do not
change, only where the numbers are fetched from.
