# Evaluation summary

This report records the verified local reference run on the deterministic synthetic dataset. It is evidence that the repository works; it is not a claim about performance on production data.

## Holdout design

- 72 monthly periods across 3 segments and 2 products
- Training and validation end: December 2023
- Untouched holdout: January–December 2024 (72 rows)
- Candidate selection: chronological validation period
- Selected candidate: target-standardised Elastic Net
- Comparator: previous-period persistence forecast

## Results

| Target | Model MAE | Naive MAE | MAE improvement | Model RMSE | Model R² |
|---|---:|---:|---:|---:|---:|
| Gross yield | 0.001643 | 0.001744 | 5.8% | 0.002083 | 0.922 |
| Strategic score | 1.335 | 1.661 | 19.6% | 1.797 | 0.590 |

Both KPIs beat the declared naive baseline on the final holdout. Yield error is expressed as a decimal rate: MAE `0.001643` is approximately 16.4 basis points.

## Decision scenario check

Scenario `DEC-2025-014` reduces fees by 4 basis points, raises digital engagement by 8%, and adds £8m of net flows for consumer and commercial segments from January 2024.

| Measure | Result |
|---|---:|
| Affected forecast rows | 48 |
| Mean gross-yield delta | +0.000363 (about +3.63 bps) |
| Mean strategic-score delta | +2.208 |

These are conditional model sensitivities, not causal estimates.

## Monitoring check

The reference/current comparison returns `warning` with maximum PSI `0.177`, driven by digital engagement and unemployment. That is expected because the current population contains the later synthetic time period while the reference contains training/validation data. It demonstrates the alert path without breaching the critical threshold of `0.25`.

## Verification

- 6 unit/integration tests passed
- Full data → train → evaluate → reforecast → monitor workflow completed
- Python source compilation completed
- Model artifact reload verified by integration test

