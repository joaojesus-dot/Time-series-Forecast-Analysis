from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from forecasting_multivariate import run_multivariate_analysis
    from forecasting_univariate import run_univariate_analysis
except ImportError:  # pragma: no cover - package import path
    from .forecasting_multivariate import run_multivariate_analysis
    from .forecasting_univariate import run_univariate_analysis


STALE_ROOT_FILES = [
    "auto_arima_forecasting_analysis.md",
    "auto_arima_forecasts.csv",
    "auto_arima_metrics.csv",
    "arima_forecasting_analysis.md",
    "arima_forecasts.csv",
    "arima_metrics.csv",
    "arimax_forecasting_analysis.md",
    "arimax_forecasts.csv",
    "arimax_metrics.csv",
    "arimax_skipped.csv",
    "arima_vs_arimax_comparison.csv",
    "arima_vs_arimax_comparison.md",
]

STALE_ROOT_PLOTS = [
    "auto_arima_metric_comparison.png",
    "arima_metric_comparison.png",
    "arimax_metric_comparison.png",
    "subset_B_1min_auto_arima_forecast.png",
    "subset_B_1min_arima_forecast.png",
    "subset_B_1min_arimax_forecast.png",
    "subset_B_30s_auto_arima_forecast.png",
    "subset_B_30s_arima_forecast.png",
    "subset_B_30s_arimax_forecast.png",
    "subset_B_5min_auto_arima_forecast.png",
    "subset_B_5min_arima_forecast.png",
    "subset_B_5min_arimax_forecast.png",
    "subset_B_raw_auto_arima_forecast.png",
    "subset_B_raw_arima_forecast.png",
]


def run_forecasting_pipeline(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    derived_data_dir: Path,
    timestamp_column: str = "date",
) -> dict[str, dict[str, pd.DataFrame]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    derived_data_dir.mkdir(parents=True, exist_ok=True)
    prune_forecasting_outputs(output_dir)

    results = {}
    univariate_policy = forecasting_policy.get("univariate", {})
    multivariate_policy = forecasting_policy.get("multivariate", {})

    if bool(univariate_policy.get("enabled", True)):
        results["univariate"] = run_univariate_analysis(
            datasets=datasets,
            target_column=target_column,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
            forecasting_policy=forecasting_policy,
            analysis_policy=univariate_policy,
            output_dir=output_dir / "univariate",
            reports_dir=reports_dir,
            plots_dir=plots_dir / "univariate",
            derived_data_dir=derived_data_dir,
            timestamp_column=timestamp_column,
        )

    if bool(multivariate_policy.get("enabled", False)):
        results["multivariate"] = run_multivariate_analysis(
            datasets=datasets,
            target_column=target_column,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
            forecasting_policy=forecasting_policy,
            analysis_policy=multivariate_policy,
            output_dir=output_dir / "multivariate",
            reports_dir=reports_dir,
            plots_dir=plots_dir / "multivariate",
            timestamp_column=timestamp_column,
        )

    return results


def prune_forecasting_outputs(output_dir: Path) -> None:
    for filename in STALE_ROOT_FILES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    for filename in STALE_ROOT_PLOTS:
        path = output_dir / "plots" / filename
        if path.exists():
            path.unlink()
