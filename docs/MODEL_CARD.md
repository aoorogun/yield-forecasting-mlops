# Model card

## Model details

- **Name:** Yield and Strategic Objective Forecaster
- **Version:** 1.0.0
- **Owner:** Data Science / Financial Planning
- **Model family:** selected from Elastic Net, Random Forest, and histogram gradient boosting
- **Outputs:** gross yield and strategic-objective score, with empirical prediction intervals

## Intended use

Support planning, sensitivity analysis, and decision impact assessment at monthly segment/product level. Forecasts augment, rather than replace, accountable business judgement.

## Out-of-scope use

- Individual customer pricing or eligibility decisions
- Automated trading or investment recommendations
- Stress scenarios outside observed driver ranges without expert review
- Causal claims: scenario deltas are model-based sensitivities, not proof of causal impact

## Data

This repository trains on deterministic synthetic data. A production implementation should document source ownership, reporting latency, corrections, lineage, retention, and lawful use. Protected attributes should not be used. Proxy risk should be assessed even when protected attributes are absent.

## Evaluation

Candidate selection uses a chronological validation interval. Final metrics are computed once on the subsequent 12-month holdout and compared with a previous-period naive forecast. Metrics are reported per target; a range-normalised mean MAE supports selection across target scales.

Production acceptance gates:

- Beats the naive baseline on both target MAEs
- Gross-yield MAE below the business-approved threshold
- Strategic-score error stable across segment and product slices
- Prediction interval coverage within five percentage points of its nominal level
- No critical data-quality or drift alert

## Limitations and risks

- Synthetic relationships do not establish performance on real data.
- Regime shifts can invalidate relationships learned from history.
- Missing-value assumptions may mute emerging risks; flags must be visible to consumers.
- Tree ensembles extrapolate poorly beyond training ranges.
- Scenario changes can combine in non-intuitive ways and require subject-matter review.
- Empirical residual intervals are simple and may not provide conditional coverage.

## Monitoring and governance

Track input freshness, missingness, schema validity, PSI, prediction distribution, realised MAE/RMSE, interval coverage, and segment-level errors. Critical drift stops automatic promotion and triggers triage. Every model is registered with code version, data hash, parameters, validation evidence, approver, and rollback target.

