from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
# Reports are generated from already-computed dataframes. This keeps the
# narrative layer separate from the analysis layer and prevents hidden
# recalculation inside markdown generation.

def write_documentation_outputs(
    reports_dir: Path,
    repaired_filename: str,
    target_column: str,
    feature_selection: dict[str, object],
    quality_report: str,
    resampling_policy: dict[str, object],
    granularity_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    smoothing_summary: pd.DataFrame,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "dataset_overview.md": build_dataset_overview_markdown(
            quality_report,
            repaired_filename,
            target_column,
            feature_selection,
            distribution_summary,
        ),
        "pre_forecasting_report.md": build_pre_forecasting_markdown(
            granularity_summary,
            resampling_policy,
            smoothing_summary,
        ),
    }

    for filename, content in output_files.items():
        output_path = reports_dir / filename
        output_path.write_text(content, encoding="utf-8")
        print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------
# These functions intentionally stay presentation-oriented. Detailed
# machine-readable decisions remain in config files, Excel summary workbooks,
# or large CSV datasets.

def build_dataset_overview_markdown(
    quality_report: str,
    repaired_filename: str,
    target_column: str,
    feature_selection: dict[str, object],
    distribution_summary: pd.DataFrame,
) -> str:
    target_raw = distribution_summary[
        (distribution_summary["id_key"] == "subset_B_raw") & (distribution_summary["variable"] == target_column)
    ].iloc[0].to_dict()
    return "\n".join(
        [
            "# Boiler Dataset Overview",
            "",
            "## Scope",
            "- The project is focused on the Chinese boiler dataset.",
            f"- Repaired working dataset: `{repaired_filename}`.",
            f"- Forecasting target: `{target_column}`.",
            "- Candidate B is the primary reduced forecasting dataset.",
            "- Candidate C is retained as a control-aware comparison for later modeling.",
            "",
            "## Feature Sets",
            f"- Candidate A features: `{len(feature_selection['candidate_a_features'])}`.",
            f"- Candidate B features: `{len(feature_selection['candidate_b_features'])}`.",
            f"- Candidate C features: `{len(feature_selection['candidate_c_features'])}`.",
            "- Feature selection is derived directly from `python_scripts/config/boiler_family_reduction.json`.",
            "",
            "## Raw Target",
            f"- Mean: `{target_raw['mean']:.3f}`.",
            f"- Standard deviation: `{target_raw['std']:.3f}`.",
            f"- Minimum: `{target_raw['min']:.3f}`.",
            f"- Maximum: `{target_raw['max']:.3f}`.",
            "",
            "## Source Data Quality",
            quality_report.strip(),
        ]
    )


def build_pre_forecasting_markdown(
    granularity_summary: pd.DataFrame,
    resampling_policy: dict[str, object],
    smoothing_summary: pd.DataFrame,
) -> str:
    rows = [
        f"- `{row['id_key']}`: {row['rows']:,} rows from `{row['start_date']}` to `{row['end_date']}`."
        for row in granularity_summary.to_dict("records")
    ]
    column_aggregations = resampling_policy.get("column_aggregations", {})
    column_aggregation_lines = [
        f"- `{column}`: `{aggregation}`."
        for column, aggregation in sorted(column_aggregations.items())
    ]
    std_lines = [
        f"- `{row['id_key']}` `{row['transform']}` `{row['window']}`: std `{row['std']:.3f}`."
        for row in smoothing_summary.to_dict("records")
        if row["transform"] in {"original", "trailing_rolling_mean", "first_difference", "second_difference"}
    ]
    return "\n".join(
        [
            "# Pre-Forecasting Report",
            "",
            "## Resampling",
            "- Candidate B is generated at `raw`, `30s`, `1min`, and `5min` granularities.",
            "- The raw cadence is 5 seconds.",
            "- Resampled timestamps are causal window endpoints.",
            f"- Windows are `{resampling_policy['closed']}`-closed and labeled on the `{resampling_policy['label']}` edge.",
            f"- Window origin is `{resampling_policy['origin']}`.",
            f"- Incomplete edge windows are dropped: `{resampling_policy['drop_partial_windows']}`.",
            f"- Default aggregation: `{resampling_policy['default_aggregation']}`.",
            "",
            "## Configured Column Aggregation Overrides",
            *(column_aggregation_lines or ["- None."]),
            "",
            "## Generated Modeling Datasets",
            *rows,
            "",
            "## Smoothing And Differencing",
            "- Smoothing and differencing remain diagnostic transforms for now.",
            "- The level target remains the default modeling target.",
            "- Differencing and smoothing may become modeling inputs later if validation supports them.",
            "",
            "## Transform Summary",
            *std_lines,
        ]
    )


# ---------------------------------------------------------------------------
# Forecasting report builders
# ---------------------------------------------------------------------------
# Forecasting reports summarize model results that have already been computed by
# model-specific modules. They stay here with the rest of the narrative layer.

def write_arima_report(
    metrics: pd.DataFrame,
    output_path: Path,
    order: tuple[int, int, int],
    target_granularities: list[str],
) -> None:
    p_value, d_value, q_value = order
    lines = [
        "# Univariate ARIMA Analysis",
        "",
        "## Scope",
        "- Analysis family: univariate.",
        "- Forecasting target: `TE_8313B.AV_0#`.",
        f"- Active granularities: `{', '.join(target_granularities)}`.",
        "- Chronological 80/20 split.",
        "- One forecast is produced for the full holdout period.",
        f"- Nixtla `statsforecast` ARIMA is run as fixed `ARIMA({p_value}, {d_value}, {q_value})`.",
        "",
        "## Metrics",
        "| Granularity | Model | p | d | q | MAE | RMSE | MAPE % | sMAPE % | Bias | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values(["granularity", "p", "d", "q"]).to_dict("records"):
        lines.append(format_forecast_metric_row(row))

    best = metrics.sort_values("mae").iloc[0].to_dict()
    lines.extend(
        [
            "",
            "## Initial Reading",
            f"- Lowest MAE in this run: `{best['granularity']}` `{best['model']}` with MAE `{best['mae']:.3f}`.",
            "- The raw target scaling artifact prepares the next MLP baseline.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_arimax_report(
    metrics: pd.DataFrame,
    skipped_rows: list[dict[str, object]],
    output_path: Path,
    order: tuple[int, int, int],
) -> None:
    p_value, d_value, q_value = order
    lines = [
        "# Multivariate ARIMAX Analysis",
        "",
        "## Scope",
        "- Analysis family: multivariate.",
        "- Forecasting target: `TE_8313B.AV_0#`.",
        "- Non-target Candidate B variables are used as exogenous regressors.",
        f"- Nixtla `statsforecast` ARIMAX is run as fixed `ARIMA({p_value}, {d_value}, {q_value})`.",
        "- Future test-period exogenous values are supplied for evaluation.",
        "",
        "## Metrics",
        "| Granularity | Model | p | d | q | MAE | RMSE | MAPE % | sMAPE % | Bias | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values(["granularity", "p", "d", "q"]).to_dict("records"):
        lines.append(format_forecast_metric_row(row))

    if skipped_rows:
        lines.extend(["", "## Skipped"])
        for row in skipped_rows:
            lines.append(
                f"- `{row['granularity']}` skipped: `{row['train_rows']}` train rows exceeds `{row['max_train_rows']}`."
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_univariate_forecasting_report(
    output_path: Path,
    target_column: str,
    arima_metrics: pd.DataFrame,
    scaling_summary: pd.DataFrame,
    mlp_summary: pd.DataFrame,
) -> None:
    best = arima_metrics.sort_values("mae").iloc[0].to_dict()
    scaling = scaling_summary.iloc[0].to_dict()
    mlp = mlp_summary.iloc[0].to_dict()
    lines = [
        "# Univariate Forecasting Analysis",
        "",
        "## Scope",
        f"- Target: `{target_column}`.",
        "- Active data source: `subset_B_raw` only.",
        "- Current statistical baseline: fixed-order univariate ARIMA.",
        "- Next model family prepared: MLP using scaled target lags only.",
        "",
        "## Target Scaling",
        f"- Scaler: `{scaling['scaler']}` fitted on the train split only.",
        f"- Train mean: `{scaling['train_mean']:.6f}`.",
        f"- Train std: `{scaling['train_std']:.6f}`.",
        "- Scaled series artifact: `data/chinese_boiler_dataset/derived/raw_target_scaled.csv`.",
        "",
        "## MLP Preparation",
        f"- Lookback: `{mlp['lookback_steps']}` raw steps (`{mlp['lookback_duration']}`).",
        f"- Horizon: `{mlp['horizon_steps']}` raw steps (`{mlp['horizon_duration']}`).",
        f"- Train windows available: `{mlp['train_windows']}`.",
        f"- Test windows available: `{mlp['test_windows']}`.",
        "- Window arrays are intentionally not written yet; they should be generated in memory when the MLP trainer is added.",
        "",
        "## Current Baseline",
        f"- Best raw ARIMA result: `{best['model']}` with MAE `{best['mae']:.3f}` and RMSE `{best['rmse']:.3f}`.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_forecast_metric_row(row: dict[str, object]) -> str:
    return (
        f"| {row['granularity']} | {row['model']} | {row['p']} | {row['d']} | {row['q']} | "
        f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['mape']:.3f} | "
        f"{row['smape']:.3f} | {row['bias']:.3f} | {row['warning_count']} |"
    )
