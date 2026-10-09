# Episodic Data Engine -- common tasks.
#
# Every target that touches data runs under scripts/capped, which enforces a
# cgroup memory ceiling. On WSL an OOM can take the whole VM down, so a runaway
# stage must die alone. Override with RDE_MEM_MAX (default 6G).

PY      ?= ../.venv/bin/python
RDE     ?= ../.venv/bin/rde
CAP     := ./scripts/capped
DATASET ?= so100_pickplace
TIER    ?= 0
CONFIG  ?= configs/so100_pickplace.json

.PHONY: help install test test-cov lint pipeline pipeline-aloha \
        inspect faults resources models lineage flywheel compare \
        checks versions clean-derived observability cardinality \
        obs-video obs-contract

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:         ## Install the package (editable) plus dev extras
	uv pip install -e . --extra dev

test:            ## Run the test suite
	$(PY) -m pytest tests -q

test-cov:        ## Run tests with a coverage report
	$(PY) -m pytest tests -q --cov=rde --cov-report=term-missing

pipeline:        ## Full pipeline on the SO-100 dataset (CONFIG= to override)
	$(CAP) $(RDE) run --config $(CONFIG)

pipeline-aloha:  ## Full pipeline on the bimanual 4-camera ALOHA dataset
	$(CAP) $(RDE) run --config configs/aloha_static_coffee.json

inspect:         ## Inspect the source dataset without ingesting it
	$(CAP) $(PY) examples/01_inspect_source.py

faults:          ## Prove the QC checks fire, via fault injection
	$(CAP) $(PY) examples/02_fault_injection_demo.py

resources:       ## Measure resource usage
	$(CAP) $(PY) examples/03_resource_report.py

models:          ## Demonstrate model interchangeability
	$(CAP) $(PY) examples/04_swap_models.py

lineage:         ## Answer the two provenance questions
	$(CAP) $(PY) examples/05_lineage_queries.py

flywheel:        ## Close the deployment feedback loop
	$(CAP) $(PY) examples/06_feedback_flywheel.py

compare:         ## Compare the engine across both datasets
	$(CAP) $(PY) examples/07_compare_datasets.py

observability:   ## Two end-to-end failure investigations (sensor + drift)
	$(CAP) $(PY) examples/09_observability_scenarios.py

cardinality:     ## Measure what episode_id as a metric label actually costs
	$(CAP) $(PY) examples/08_cardinality_experiment.py

obs-runs:        ## Pipeline run history (DATASET= to override)
	$(RDE) obs runs $(DATASET)

obs-health:      ## Per-episode health, worst first
	$(RDE) obs health $(DATASET) raw/v1

obs-video:       ## Video health without decoding everything (TIER=0 or 1)
	$(RDE) obs video $(DATASET) raw/v1 --tier $(TIER)

obs-contract:    ## Infer a data health contract from raw/v1
	$(CAP) $(RDE) obs contract $(DATASET) raw/v1

obs-metrics:     ## Inspect the exported Prometheus textfile
	$(RDE) obs metrics

checks:          ## List every registered validation check
	$(RDE) qc list-checks

versions:        ## List all dataset versions in the catalog
	$(RDE) version list

clean-derived:   ## Drop derived versions; keeps source cache and blobs
	@echo "Removing catalog, tables, artifacts and run records."
	@echo "Source cache and blobs are preserved."
	rm -rf $${RDE_DATA_ROOT:-$$HOME/VLAExperiments/rde_data}/catalog.db* \
	       $${RDE_DATA_ROOT:-$$HOME/VLAExperiments/rde_data}/tables \
	       $${RDE_DATA_ROOT:-$$HOME/VLAExperiments/rde_data}/artifacts \
	       $${RDE_DATA_ROOT:-$$HOME/VLAExperiments/rde_data}/runs
