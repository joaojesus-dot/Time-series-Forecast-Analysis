from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA


def prune_forecasting_outputs(output_dir: Path, forecasting_policy: dict[str, Any]) -> None:
    stale_files = [
        "auto_arima_forecasting_analysis.md",
        "auto_arima_forecasts.csv",
        "auto_arima_metrics.csv",
    ]
    stale_plots = [
        "auto_arima_metric_comparison.png",
        "subset_B_1min_auto_arima_forecast.png",
        "subset_B_30s_auto_arima_forecast.png",
        "subset_B_5min_auto_arima_forecast.png",
        "subset_B_raw_auto_arima_forecast.png",
    ]
    if not bool(forecasting_policy.get("write_full_forecasts", False)):
        stale_files.extend(["arima_forecasts.csv", "arimax_forecasts.csv"])

    for filename in stale_files:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    for filename in stale_plots:
        path = output_dir / "plots" / filename
        if path.exists():
            path.unlink()


def run_univariate_arima_forecasts(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    output_dir: Path,
    timestamp_column: str = "date",
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_full_forecasts = bool(forecasting_policy.get("write_full_forecasts", False))
    write_forecast_plots = bool(forecasting_policy.get("write_forecast_plots", True))

    forecast_frames = []
    metric_rows = []
    arima_config = forecasting_policy["arima"]
    p_value = int(arima_config["p"])
    q_value = int(arima_config["q"])

    for granularity in forecasting_policy["target_granularities"]:
        id_key = f"subset_B_{granularity}"
        if id_key not in datasets:
            continue

        series_frame = build_statsforecast_frame(datasets[id_key], id_key, target_column, timestamp_column)
        train_frame, test_frame = split_train_test(series_frame, float(forecasting_policy["train_fraction"]))
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        granularity_forecasts = []

        for d_value in forecasting_policy["arima_d_values"]:
            model_alias = f"ARIMA_p{p_value}_d{d_value}_q{q_value}"
            forecast_frame = forecast_arima(
                train_frame=train_frame,
                test_frame=test_frame,
                frequency=frequency,
                p_value=p_value,
                d_value=int(d_value),
                q_value=q_value,
                model_alias=model_alias,
                arima_config=arima_config,
            )
            forecast_frame["id_key"] = id_key
            forecast_frame["granularity"] = granularity
            forecast_frame["model"] = model_alias
            forecast_frame["p"] = p_value
            forecast_frame["d"] = int(d_value)
            forecast_frame["q"] = q_value
            forecast_frame["train_rows"] = len(train_frame)
            forecast_frame["test_rows"] = len(test_frame)
            forecast_frames.append(forecast_frame)
            granularity_forecasts.append(forecast_frame)

            metric_rows.append(
                {
                    "id_key": id_key,
                    "granularity": granularity,
                    "model": model_alias,
                    "p": p_value,
                    "d": int(d_value),
                    "q": q_value,
                    "train_rows": len(train_frame),
                    "test_rows": len(test_frame),
                    "warning_count": int(forecast_frame.attrs.get("warning_count", 0)),
                    "warnings": "; ".join(forecast_frame.attrs.get("warnings", [])),
                    **forecast_metrics(forecast_frame["y"], forecast_frame["forecast"]),
                }
            )

        if write_forecast_plots:
            write_forecast_plot(id_key, granularity, train_frame, test_frame, granularity_forecasts, plots_dir)

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["granularity", "d"])
    if write_full_forecasts:
        write_csv_output(forecasts, output_dir / "arima_forecasts.csv")
    write_csv_output(metrics, output_dir / "arima_metrics.csv")
    write_metrics_plot(metrics, plots_dir / "arima_metric_comparison.png")
    write_forecast_report(metrics, output_dir / "arima_forecasting_analysis.md", p_value, q_value)
    return {"forecasts": forecasts, "metrics": metrics}


def run_arimax_forecasts(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    output_dir: Path,
    timestamp_column: str = "date",
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_full_forecasts = bool(forecasting_policy.get("write_full_forecasts", False))
    write_forecast_plots = bool(forecasting_policy.get("write_forecast_plots", True))

    forecast_frames = []
    metric_rows = []
    arima_config = forecasting_policy["arima"]
    arimax_config = forecasting_policy["arimax"]
    p_value = int(arima_config["p"])
    q_value = int(arima_config["q"])
    max_train_rows = int(arimax_config.get("max_train_rows", 12_000))
    skipped_rows = []

    for granularity in arimax_config.get("target_granularities", forecasting_policy["target_granularities"]):
        id_key = f"subset_B_{granularity}"
        if id_key not in datasets:
            continue

        exogenous_columns = infer_exogenous_columns(datasets[id_key], target_column, timestamp_column)
        series_frame = build_statsforecast_frame(
            datasets[id_key],
            id_key,
            target_column,
            timestamp_column,
            exogenous_columns=exogenous_columns,
        )
        train_frame, test_frame = split_train_test(series_frame, float(forecasting_policy["train_fraction"]))
        if len(train_frame) > max_train_rows:
            skipped_rows.append(
                {
                    "id_key": id_key,
                    "granularity": granularity,
                    "train_rows": len(train_frame),
                    "max_train_rows": max_train_rows,
                    "reason": "Skipped because statsforecast ARIMA with exogenous regressors performs a dense SVD that is not memory-safe at this row count.",
                }
            )
            continue
        if bool(arimax_config.get("standardize_exogenous", True)):
            train_frame, test_frame = standardize_exogenous(train_frame, test_frame, exogenous_columns)
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        granularity_forecasts = []

        for d_value in forecasting_policy["arima_d_values"]:
            model_alias = f"ARIMAX_p{p_value}_d{d_value}_q{q_value}"
            forecast_frame = forecast_arima(
                train_frame=train_frame,
                test_frame=test_frame,
                frequency=frequency,
                p_value=p_value,
                d_value=int(d_value),
                q_value=q_value,
                model_alias=model_alias,
                arima_config=arima_config,
                exogenous_columns=exogenous_columns,
            )
            forecast_frame["id_key"] = id_key
            forecast_frame["granularity"] = granularity
            forecast_frame["model"] = model_alias
            forecast_frame["p"] = p_value
            forecast_frame["d"] = int(d_value)
            forecast_frame["q"] = q_value
            forecast_frame["exogenous_count"] = len(exogenous_columns)
            forecast_frame["train_rows"] = len(train_frame)
            forecast_frame["test_rows"] = len(test_frame)
            forecast_frames.append(forecast_frame)
            granularity_forecasts.append(forecast_frame)

            metric_rows.append(
                {
                    "id_key": id_key,
                    "granularity": granularity,
                    "model": model_alias,
                    "p": p_value,
                    "d": int(d_value),
                    "q": q_value,
                    "exogenous_count": len(exogenous_columns),
                    "exogenous_columns": "; ".join(exogenous_columns),
                    "train_rows": len(train_frame),
                    "test_rows": len(test_frame),
                    "warning_count": int(forecast_frame.attrs.get("warning_count", 0)),
                    "warnings": "; ".join(forecast_frame.attrs.get("warnings", [])),
                    **forecast_metrics(forecast_frame["y"], forecast_frame["forecast"]),
                }
            )

        if write_forecast_plots:
            write_forecast_plot(
                id_key,
                granularity,
                train_frame,
                test_frame,
                granularity_forecasts,
                plots_dir,
                model_family="ARIMAX",
            )

    if skipped_rows:
        write_csv_output(pd.DataFrame(skipped_rows), output_dir / "arimax_skipped.csv")
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["granularity", "d"])
    if write_full_forecasts:
        write_csv_output(forecasts, output_dir / "arimax_forecasts.csv")
    write_csv_output(metrics, output_dir / "arimax_metrics.csv")
    write_metrics_plot(metrics, plots_dir / "arimax_metric_comparison.png")
    write_forecast_report(
        metrics,
        output_dir / "arimax_forecasting_analysis.md",
        p_value,
        q_value,
        model_family="ARIMAX",
        extra_scope_lines=[
            "- Multivariate run: the non-target Candidate B variables are used as exogenous regressors.",
            "- Future test-period exogenous values are supplied to `statsforecast` for evaluation.",
            "- Exogenous regressors are standardized using training-split statistics only.",
            f"- Granularities above `{max_train_rows}` training rows are skipped for this API to avoid dense-SVD memory failures.",
            *[
                f"- Skipped `{row['granularity']}` ARIMAX: `{row['train_rows']}` training rows exceeds `{row['max_train_rows']}`."
                for row in skipped_rows
            ],
        ],
    )
    return {"forecasts": forecasts, "metrics": metrics}


def write_arima_arimax_comparison(
    arima_metrics: pd.DataFrame,
    arimax_metrics: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    left = arima_metrics[
        ["granularity", "p", "d", "q", "model", "mae", "rmse", "mape", "smape", "bias"]
    ].rename(
        columns={
            "model": "arima_model",
            "mae": "arima_mae",
            "rmse": "arima_rmse",
            "mape": "arima_mape",
            "smape": "arima_smape",
            "bias": "arima_bias",
        }
    )
    right = arimax_metrics[
        ["granularity", "p", "d", "q", "model", "exogenous_count", "mae", "rmse", "mape", "smape", "bias"]
    ].rename(
        columns={
            "model": "arimax_model",
            "mae": "arimax_mae",
            "rmse": "arimax_rmse",
            "mape": "arimax_mape",
            "smape": "arimax_smape",
            "bias": "arimax_bias",
        }
    )
    comparison = left.merge(right, on=["granularity", "p", "d", "q"], how="inner")
    comparison["mae_delta"] = comparison["arimax_mae"] - comparison["arima_mae"]
    comparison["mae_change_percent"] = (comparison["mae_delta"] / comparison["arima_mae"]) * 100
    comparison["rmse_delta"] = comparison["arimax_rmse"] - comparison["arima_rmse"]
    comparison["rmse_change_percent"] = (comparison["rmse_delta"] / comparison["arima_rmse"]) * 100
    comparison = comparison.sort_values(["granularity", "d"])

    write_csv_output(comparison, output_dir / "arima_vs_arimax_comparison.csv")
    write_arima_arimax_comparison_report(comparison, output_dir / "arima_vs_arimax_comparison.md")
    return comparison


def write_arima_arimax_comparison_report(comparison: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# ARIMA vs ARIMAX Comparison",
        "",
        "## Scope",
        "- Compares fixed-order univariate ARIMA against fixed-order ARIMAX on overlapping granularities.",
        "- Negative delta means ARIMAX improved the metric.",
        "- ARIMAX uses observed future exogenous values from the test period, so this is an oracle-style evaluation unless those variables are known or forecasted at prediction time.",
        "",
        "## Metrics",
        "| Granularity | d | ARIMA MAE | ARIMAX MAE | MAE Change % | ARIMA RMSE | ARIMAX RMSE | RMSE Change % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict("records"):
        lines.append(
            f"| {row['granularity']} | {row['d']} | {row['arima_mae']:.3f} | {row['arimax_mae']:.3f} | "
            f"{row['mae_change_percent']:.1f} | {row['arima_rmse']:.3f} | {row['arimax_rmse']:.3f} | "
            f"{row['rmse_change_percent']:.1f} |"
        )

    best = comparison.sort_values("mae_delta").iloc[0].to_dict()
    lines.extend(
        [
            "",
            "## Initial Reading",
            f"- Largest MAE improvement: `{best['granularity']}` with `d={best['d']}`, "
            f"from `{best['arima_mae']:.3f}` to `{best['arimax_mae']:.3f}`.",
            "- `d=0` remains the best differencing choice among the tested ARIMAX models.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_statsforecast_frame(
    dataset: pd.DataFrame,
    id_key: str,
    target_column: str,
    timestamp_column: str,
    exogenous_columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_columns = [timestamp_column, target_column] + list(exogenous_columns or [])
    frame = dataset[selected_columns].copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    for column in exogenous_columns or []:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=selected_columns).sort_values(timestamp_column)
    frame = frame.rename(columns={timestamp_column: "ds", target_column: "y"}).assign(unique_id=id_key)
    return frame[["unique_id", "ds", "y"] + list(exogenous_columns or [])]


def infer_exogenous_columns(dataset: pd.DataFrame, target_column: str, timestamp_column: str) -> list[str]:
    excluded = {target_column, timestamp_column}
    return [
        column
        for column in dataset.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def standardize_exogenous(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    exogenous_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = train_frame.copy()
    test_frame = test_frame.copy()
    for column in exogenous_columns:
        mean = train_frame[column].mean()
        std = train_frame[column].std(ddof=0)
        if pd.isna(std) or std == 0:
            std = 1.0
        train_frame[column] = (train_frame[column] - mean) / std
        test_frame[column] = (test_frame[column] - mean) / std
    return train_frame, test_frame


def split_train_test(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    split_index = int(len(frame) * train_fraction)
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("train_fraction creates an empty train or test set.")
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def statsforecast_frequency(
    granularity: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
) -> str:
    if granularity == "raw":
        return str(resampling_policy["input_frequency"])
    frequency = granularity_options.get(granularity)
    if frequency is None:
        raise ValueError(f"Could not infer frequency for granularity '{granularity}'.")
    return str(frequency)


def forecast_arima(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    frequency: str,
    p_value: int,
    d_value: int,
    q_value: int,
    model_alias: str,
    arima_config: dict[str, Any],
    exogenous_columns: list[str] | None = None,
) -> pd.DataFrame:
    model = ARIMA(
        order=(p_value, d_value, q_value),
        season_length=int(arima_config.get("season_length", 1)),
        seasonal_order=tuple(int(value) for value in arima_config.get("seasonal_order", [0, 0, 0])),
        include_mean=d_value == 0,
        include_drift=d_value == 1,
        method=str(arima_config.get("method", "CSS-ML")),
        alias=model_alias,
    )
    forecast_engine = StatsForecast(models=[model], freq=frequency, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        future_exogenous = None
        if exogenous_columns:
            future_exogenous = test_frame[["unique_id", "ds"] + exogenous_columns]
        forecast = forecast_engine.forecast(df=train_frame, h=len(test_frame), X_df=future_exogenous)
    forecast = forecast[["ds", model_alias]].rename(columns={model_alias: "forecast"})
    evaluated = test_frame[["ds", "y"]].merge(forecast, on="ds", how="inner")
    if len(evaluated) != len(test_frame):
        raise ValueError(
            f"Forecast/test timestamp alignment failed for {model_alias}: "
            f"{len(evaluated)} aligned rows out of {len(test_frame)} test rows."
        )
    warning_messages = sorted({str(item.message) for item in caught_warnings})
    evaluated.attrs["warning_count"] = len(caught_warnings)
    evaluated.attrs["warnings"] = warning_messages
    return evaluated


def forecast_metrics(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    errors = forecast - actual
    abs_errors = errors.abs()
    nonzero_actual = actual.replace(0, np.nan)
    denominator = (actual.abs() + forecast.abs()).replace(0, np.nan)
    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float((abs_errors / nonzero_actual.abs()).mean() * 100),
        "smape": float((2 * abs_errors / denominator).mean() * 100),
        "bias": float(errors.mean()),
    }


def write_forecast_plot(
    id_key: str,
    granularity: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    forecast_frames: list[pd.DataFrame],
    output_dir: Path,
    model_family: str = "ARIMA",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    train_tail = train_frame.tail(min(len(train_frame), len(test_frame)))

    axes[0].plot(train_tail["ds"], train_tail["y"], linewidth=0.9, color="#888888", label="train tail")
    axes[0].plot(test_frame["ds"], test_frame["y"], linewidth=1.0, color="#222222", label="test actual")
    for forecast_frame in forecast_frames:
        axes[0].plot(
            forecast_frame["ds"],
            forecast_frame["forecast"],
            linewidth=1.0,
            label=str(forecast_frame["model"].iloc[0]),
        )
    axes[0].axvline(test_frame["ds"].iloc[0], color="#333333", linewidth=0.9, linestyle="--")
    axes[0].set_title(f"{id_key} - 80/20 {model_family} Forecast")
    axes[0].set_ylabel("Target")
    axes[0].legend(loc="upper right", fontsize=8)

    for forecast_frame in forecast_frames:
        axes[1].plot(
            forecast_frame["ds"],
            forecast_frame["forecast"] - forecast_frame["y"],
            linewidth=1.0,
            label=str(forecast_frame["model"].iloc[0]),
        )
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Forecast Error")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel("Forecast - actual")
    axes[1].legend(loc="upper right", fontsize=8)
    save_figure(fig, output_dir / f"{id_key}_{model_family.lower()}_forecast.png")


def write_metrics_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metric in zip(axes, ["mae", "rmse"]):
        pivot = metrics.pivot(index="granularity", columns="model", values=metric)
        pivot = pivot.reindex(["raw", "30s", "1min", "5min"])
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(metric.upper())
        axis.set_xlabel("Granularity")
        axis.set_ylabel(metric.upper())
        axis.legend(loc="upper right", fontsize=8)
    save_figure(fig, output_path)


def write_forecast_report(
    metrics: pd.DataFrame,
    output_path: Path,
    p_value: int,
    q_value: int,
    model_family: str = "ARIMA",
    extra_scope_lines: list[str] | None = None,
) -> None:
    lines = [
        f"# {model_family} Forecasting Analysis",
        "",
        "## Scope",
        "- Forecasting target: `TE_8313B.AV_0#`.",
        "- Chronological 80/20 split.",
        "- One forecast is produced for the full holdout period of each granularity.",
        f"- Nixtla `statsforecast` {model_family} is run as fixed `ARIMA({p_value}, d, {q_value})`.",
        "- Tested differencing orders: `d=0`, `d=1`, and `d=2`; `d=2` is the second-difference case.",
    ]
    if extra_scope_lines:
        lines.extend(extra_scope_lines)
    lines.extend(
        [
            "",
            "## Metrics",
            "| Granularity | Model | p | d | q | MAE | RMSE | MAPE % | sMAPE % | Bias | Warnings |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered_metrics = metrics.copy()
    ordered_metrics["granularity"] = pd.Categorical(
        ordered_metrics["granularity"],
        categories=["raw", "30s", "1min", "5min"],
        ordered=True,
    )
    for row in ordered_metrics.sort_values(["granularity", "p", "d", "q"]).to_dict("records"):
        lines.append(
            f"| {row['granularity']} | {row['model']} | {row['p']} | {row['d']} | {row['q']} | "
            f"{row['mae']:.3f} | {row['rmse']:.3f} | "
            f"{row['mape']:.3f} | {row['smape']:.3f} | {row['bias']:.3f} | {row['warning_count']} |"
        )

    best = metrics.sort_values("mae").iloc[0].to_dict()
    lines.extend(
        [
            "",
            "## Initial Reading",
            f"- Lowest MAE in this run: `{best['granularity']}` `{best['model']}` with MAE `{best['mae']:.3f}`.",
            "- These are initial holdout results, not yet rolling-origin validation.",
            "- Convergence warnings are counted in the metrics table and should be reviewed before treating a model as final.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_output(frame: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
