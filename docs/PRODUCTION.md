# Production deployment and operations

## Deployment sequence

1. CI runs linting, unit tests, the full pipeline, and artifact validation.
2. A managed training job writes versioned artifacts to encrypted S3 and records lineage.
3. The evaluation step compares the challenger against the naive forecast and current champion.
4. Passing models enter SageMaker Model Registry as `PendingManualApproval`.
5. An authorised approver reviews the model card, slice metrics, drift, and scenario sanity tests.
6. Staging receives the exact approved container digest and model package.
7. After smoke and load tests, production uses a canary with 10% traffic before full promotion.
8. Rollback points to the previous endpoint configuration; artifacts are never overwritten.

## Security controls

- GitHub authenticates to AWS using short-lived OIDC credentials—no static AWS keys.
- The SageMaker role is least-privilege and scoped to the artifact bucket.
- S3 versioning, KMS encryption, public-access blocking, CloudTrail, and VPC endpoints are enabled.
- Endpoint subnets are private; security groups allow only the service layer.
- Decision payloads and forecasts are encrypted in transit and access logged.
- Image and dependency scanning block critical vulnerabilities.

The CloudFormation file is a minimal baseline. Production stacks should also supply VPC, KMS key policy, ECR repository, endpoint, WAF/API layer, notification topic, and organisation-specific tagging.

## SLOs

| Measure | Target |
|---|---:|
| Online availability | 99.9% monthly |
| p95 latency, batches <= 100 | < 500 ms |
| Scheduled forecast delivered | before 07:00 business time |
| Data freshness | source month + 1 business day |
| Critical alert acknowledgement | < 30 minutes |

## Retraining policy

Retrain monthly after source close, or on demand after critical drift, material policy changes, or two consecutive breached accuracy windows. Automated retraining does not mean automated promotion: quality gates and approval remain mandatory.

## Incident runbook

1. Confirm whether the alert is data quality, drift, service reliability, or realised performance.
2. Freeze model promotion and retain the last approved forecast.
3. Compare source counts, missingness, ranges, and upstream release notes against the previous good batch.
4. For service faults, roll back the endpoint configuration and verify health and prediction smoke tests.
5. For data faults, quarantine the batch and request a corrected source; do not silently widen assumptions.
6. For genuine drift, run challenger evaluation and segment diagnostics before approval.
7. Record timeline, impact, decisions, and preventive actions.

## Dynamic decision controls

Each decision carries an immutable ID, owner, status, effective period, population filter, changes, and approval record. Proposed decisions are visually distinct from committed decisions. Baseline and scenario must use the same model version and source snapshot. The service returns this lineage with the impact result in a production implementation.

