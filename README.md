# Yield Forecasting & Dynamic Reforecasting Platform

An end-to-end reference implementation for predicting portfolio yield and a strategic-objective KPI from financial and operational drivers. It includes leakage-safe feature engineering, supervised and unsupervised learning, a PyTorch model, assumption management for unreported periods, decision-scenario reforecasting, monitoring, a FastAPI service, and AWS SageMaker deployment assets.

> Portfolio note: all records are synthetic. They model plausible business relationships but contain no customer, employer, or confidential data.

## What this demonstrates

| Job requirement | Evidence in this repository |
|---|---|
| Multivariate predictive ML | Elastic Net, Random Forest, HistGradientBoosting, Isolation Forest, and a PyTorch MLP |
| Yield and non-yield KPIs | Joint forecasts for `gross_yield` and `strategic_score` |
| Assumptions management | Segment medians and forward-fill rules for non-reported driver values |
| Dynamic decision reforecast | Scenario engine returns baseline, scenario, and incremental impact |
| Production ML pipelines | Versioned artifacts, schema checks, time splits, API, container, monitoring |
| AWS / SageMaker | Training launcher, inference contract, CloudFormation, and deployment workflow |
| Scalable deployment | Stateless batch API, Docker, health endpoints, and autoscaling-ready infrastructure |

## Architecture

```text
synthetic source -> validation -> assumptions -> lag/rolling features -> time split
                                                               |
                                            candidate model comparison
                                                               |
                        model + metadata + reference distribution artifacts
                              |                    |
                         FastAPI service      drift monitoring
                              |
                 baseline forecast + decision overlay -> impact view
```

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

yield-forecast generate-data
yield-forecast train
yield-forecast evaluate
yield-forecast reforecast --scenario configs/scenario_example.json
yield-forecast monitor
```

Run the API:

```bash
uvicorn yield_forecasting.api.app:app --host 0.0.0.0 --port 8080
```

Run tests without extra test dependencies:

```bash
python -m unittest discover -s tests -v
```

Or run everything locally:

```bash
make all
```

## Outputs

Training writes the following to `artifacts/`:

- `model.joblib`: fitted multi-output forecasting pipeline
- `metadata.json`: model/version, features, split boundaries, and metrics
- `reference.csv`: reference feature distribution used for drift detection
- `test_predictions.csv`: holdout predictions for audit and visualisation

Dynamic reforecasting writes a row-level impact file to `reports/scenario_impact.csv`. Positive and negative deltas are always expressed as **scenario minus baseline**.

## Modelling choices

- The split is chronological, never random.
- Lag and rolling features use only values known before each forecast timestamp.
- Model selection uses the validation period; the test period remains untouched until final evaluation.
- A naive previous-period forecast is reported so model lift is explicit.
- Prediction intervals use empirical residual quantiles from validation data. For regulated or high-risk use, replace this with conformal or probabilistic forecasting and validate coverage by segment.
- Category handling and scaling live inside the persisted pipeline, keeping training and serving transformations identical.

## Production path

See [docs/PRODUCTION.md](docs/PRODUCTION.md) for the AWS topology, security controls, retraining policy, rollback strategy, and operational SLOs. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for intended use, limitations, and governance.

## Repository map

```text
src/yield_forecasting/   application and modelling code
tests/                   unit and integration tests
configs/                 project and decision scenario configuration
scripts/                 local and SageMaker entry points
infra/                   AWS CloudFormation reference stack
docs/                    architecture, model card, and runbook
.github/workflows/       CI and controlled deployment workflow
```

## Reproducibility

The generator, models, and scenario engine use fixed seeds. Configuration is externalised in `configs/base.yaml`; metadata captures the configuration and source-data fingerprint. The code intentionally avoids committing generated data and trained binaries.

