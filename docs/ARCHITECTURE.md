# Solution architecture

## Business flow

The platform produces monthly forecasts for portfolio yield and a strategic-objective score. Financial variables (rates, fees, assets, flows, duration, and spreads) are combined with non-financial variables (digital engagement, segment, product, and market context). Unreported inputs are replaced under explicit assumption rules and individually flagged so consumers can distinguish observed from assumed forecasts.

A decision can be submitted as proposed or committed. The scenario engine selects its effective population, changes only authorised drivers, scores the unchanged baseline and changed scenario with the same model, and exposes the difference. This isolates estimated decision impact from unrelated model or data changes.

## AWS reference topology

```text
EventBridge schedule / upstream event
                |
        SageMaker Pipeline
      /       |          \
 Processing  Training   Evaluation
      |          |          |
 versioned S3  Experiments  quality gate
                            |
                    Model Registry approval
                            |
                blue/green SageMaker endpoint
                    /                 \
              API Gateway          Batch Transform
                    |
             Decision Execution UI

Feature/quality metrics -> Model Monitor -> CloudWatch alarms -> on-call
```

## Key design decisions

1. **Panel forecasting over pure sequence modelling.** The dataset is modest and contains high-value business covariates. Tree ensembles provide a strong, inspectable baseline. The PyTorch module is retained for controlled experiments when volume justifies it.
2. **Time-based evaluation.** Random folds would train on future regimes and overstate accuracy. The most recent 12 months form the untouched test set.
3. **Train/serve parity.** Encoding, imputation, and scaling are part of one serialised pipeline. Historical aggregates should be materialised in a feature store for online use.
4. **Stateless scenario scoring.** Decisions are input data, not model mutations. That makes results repeatable and permits concurrent scenarios.
5. **Immutable promotion.** CI records commit SHA, data hash, metrics, model package, and container digest. Production deployment requires Model Registry approval.

## Data contracts

- Unique key: `date + segment + product`
- Frequency: calendar month start
- Yield: decimal rate, not percentage points
- Strategic score: continuous business score (nominally 0–100; not hard-clipped during training)
- Assumption flags: `1` where a supplied driver was missing before imputation
- Decision delta: scenario prediction minus baseline prediction

Schema, range, freshness, and key checks should run before feature computation. In production, invalid batches are quarantined; the last approved forecast stays available.

## Scaling

Offline scoring partitions data by forecast date and segment. SageMaker Processing and Batch Transform handle full reforecasts. The API caps synchronous batches at 10,000 records; larger requests go to an asynchronous queue or Batch Transform. Endpoint autoscaling should track invocations per instance and p95 latency.

