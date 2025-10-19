# Interview walkthrough

## Two-minute narrative

“I built this project around the core business need: forecast both a financial KPI—gross yield—and a non-financial strategic score, then immediately show how a proposed decision changes those forecasts. I used monthly panel data because it reflects multiple portfolio segments and products. The evaluation is chronological, the baseline is explicit, and all transformations are persisted with the model. Missing reported values have controlled assumptions and flags rather than invisible imputation. The decision engine scores baseline and scenario with the same model, so the delta is attributable to the inputs changed. I also included an optional PyTorch multi-task model, API and SageMaker serving contracts, infrastructure, CI/CD, drift monitoring, governance, and rollback.”

## Likely questions

**Why not choose deep learning by default?**  
Tabular tree models are often stronger on modest structured datasets, cheaper to operate, and easier to diagnose. The PyTorch path becomes relevant with substantially more observations, richer temporal signals, or embeddings. Model family should be earned by validation performance, not selected for novelty.

**How do you prevent leakage?**  
Dates are split chronologically. Target lags and rolling means are shifted before aggregation, so the current target never enters its own features. Preprocessing is fitted on training data inside the pipeline.

**Is a decision delta causal?**  
No. It is a conditional model sensitivity. Causal language requires identification assumptions, experiments, natural experiments, or a defensible causal design.

**How are non-reported items managed?**  
The pipeline uses the latest known value within each segment/product series and falls back to segment/global medians. Every imputed field has an assumption flag. Production would version assumptions, display them to users, and measure forecast sensitivity.

**How would you productionise this on AWS?**  
Use SageMaker Pipelines for processing/training/evaluation, Experiments and Model Registry for lineage and approval, S3/KMS for artifacts, a private endpoint or Batch Transform for inference, and Model Monitor plus CloudWatch for alerts. CI assumes an AWS role with OIDC and deploys immutable versions through staging and canary production.

## Sensible extensions

- Walk-forward cross-validation and Bayesian hyperparameter optimisation
- Conformal intervals with segment-conditional calibration
- SHAP explanations and monotonic constraints for selected drivers
- Forecast reconciliation across portfolio hierarchy
- Champion/challenger shadow deployment
- Feature Store point-in-time joins and event-driven decision scoring

