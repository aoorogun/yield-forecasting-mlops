"""Launch a managed SageMaker training job using the sklearn framework container."""

from __future__ import annotations

import os


def main() -> None:
    try:
        from sagemaker.sklearn.estimator import SKLearn

        import sagemaker
    except ImportError as exc:
        raise SystemExit("Install AWS dependencies with: pip install -e '.[aws]'") from exc
    role = os.environ["SAGEMAKER_ROLE_ARN"]
    session = sagemaker.Session()
    estimator = SKLearn(
        entry_point="scripts/run_pipeline.py",
        source_dir=".",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        base_job_name="yield-forecasting",
        sagemaker_session=session,
        environment={"YF_CONFIG": "configs/base.yaml"},
    )
    estimator.fit(wait=True, logs=True)


if __name__ == "__main__":
    main()
