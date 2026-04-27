from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


class FeatureSelection(TypedDict):
    target_column: str
    candidate_a_features: list[str]
    candidate_b_features: list[str]
    candidate_c_features: list[str]


def write_documentation_outputs(
    reports_dir: Path,
    repaired_filename: str,
    target_column: str,
    feature_selection: FeatureSelection,
    quality_report: str,
    quality_summary: dict[str, Any],
    resampling_policy: dict[str, Any],
    scaling_method: str,
    granularity_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    smoothing_summary: pd.DataFrame,
    autocorrelation_summary: pd.DataFrame,
    stationarity_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
) -> None:
    """Write the reviewer-facing profiling and preparation reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "dataset_overview.md": build_dataset_overview_markdown(
            quality_report,
            repaired_filename,
            target_column,
            feature_selection,
            distribution_summary,
        ),
        "target_profiling_report.md": build_target_profiling_markdown(
            target_column,
            granularity_summary,
            autocorrelation_summary,
            stationarity_summary,
            correlation_summary,
        ),
        "data_preparation_report.md": build_data_preparation_markdown(
            quality_summary,
            resampling_policy,
            scaling_method,
            granularity_summary,
            smoothing_summary,
            stationarity_summary,
        ),
    }

    for filename, content in output_files.items():
        (reports_dir / filename).write_text(content, encoding="utf-8")


def write_experiment_plan(
    output_path: Path,
    target_column: str,
    forecasting_policy: dict[str, Any],
    resampling_policy: dict[str, Any],
) -> None:
    """Write the active forecasting protocol in assignment language."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = forecasting_policy["experimental_protocol"]
    splits = protocol["splits"]
    target_granularities = ", ".join(f"`{value}`" for value in protocol["target_granularities"])
    difference_orders = ", ".join(str(value) for value in protocol.get("differentiation_orders", []))
    smoothing_windows = ", ".join(f"`{value}`" for value in protocol.get("train_only_smoothing_windows", []))
    mlp_config = forecasting_policy.get("univariate", {}).get("mlp", {})
    lines = [
        "# Experiment Plan",
        "",
        "## Scope",
        f"- Target variable: `{target_column}`.",
        f"- Raw input cadence: `{resampling_policy.get('input_frequency', 'unknown')}`.",
        f"- Modeled aggregation levels: {target_granularities}.",
        "",
        "## Chronological Splits",
        f"- Train: `{float(splits['train']):.0%}` of rows.",
        f"- Validation: `{float(splits['validation']):.0%}` of rows.",
        f"- Test: `{float(splits['test']):.0%}` of rows.",
        "- Validation and test remain strictly after train in time order.",
        "",
        "## Preparation Candidates",
        f"- Scaling method: `{protocol['scaling']['method']}` fitted on train only.",
        f"- Differentiation orders tested: `{difference_orders}`.",
        f"- Train-only smoothing windows tested: {smoothing_windows}.",
        "- Candidate recommendation is ranked by validation metrics before test comparison.",
        "",
        "## MLP Rules",
        f"- Lookback window: `{protocol['lookback_window']}`.",
        f"- Forecast horizon: `{protocol['forecast_horizon']}`.",
        "- Input features: target lags only.",
        "- Hidden architecture: single hidden layer.",
        f"- Activation: `{mlp_config.get('activation', 'relu')}`.",
        f"- Solver: `{mlp_config.get('solver', 'sgd')}`.",
        f"- Learning-rate policy: `{mlp_config.get('learning_rate', 'adaptive')}`.",
        f"- Initial learning-rate grid: `{json.dumps(mlp_config.get('initial_learning_rate_grid', []))}`.",
        f"- Max iterations: `{mlp_config.get('max_iter', 5000)}`.",
        f"- Recommendation mode: `{mlp_config.get('selection_mode', 'validation_recommendation')}`.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_forecasting_report_suite(
    reports_dir: Path,
    target_column: str,
    forecasting_policy: dict[str, Any],
    forecasting_results: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Write compact forecasting evaluation reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = reports_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prune_stale_forecasting_reports(reports_dir, models_dir)

    comparison_frame = build_forecasting_comparison_frame(forecasting_results)
    catalog_frame = build_forecasting_model_catalog(forecasting_results)
    model_specs = collect_forecasting_model_specs(target_column, forecasting_policy, forecasting_results)

    for spec in model_specs:
        (models_dir / f"{spec['model_key']}.md").write_text(build_model_report_markdown(spec), encoding="utf-8")

    (reports_dir / "overview.md").write_text(
        build_forecasting_overview_markdown(target_column, forecasting_policy, catalog_frame, comparison_frame),
        encoding="utf-8",
    )
    (reports_dir / "model_comparison.md").write_text(
        build_model_comparison_markdown(comparison_frame),
        encoding="utf-8",
    )


def prune_stale_forecasting_reports(reports_dir: Path, models_dir: Path) -> None:
    stale_root_files = ["run_manifest.json", "run_manifest.md", "arima_forecasting.md", "univariate_forecasting.md"]
    for filename in stale_root_files:
        path = reports_dir / filename
        if path.exists():
            path.unlink()
    for path in models_dir.glob("*.md"):
        path.unlink()


def build_dataset_overview_markdown(
    quality_report: str,
    repaired_filename: str,
    target_column: str,
    feature_selection: FeatureSelection,
    distribution_summary: pd.DataFrame,
) -> str:
    target_raw = distribution_summary[
        (distribution_summary["id_key"] == "subset_B_raw") & (distribution_summary["variable"] == target_column)
    ].iloc[0].to_dict()
    return "\n".join(
        [
            "# Dataset Overview",
            "",
            "## Scope",
            "- Dataset: Chinese boiler process monitoring data.",
            f"- Repaired working dataset: `{repaired_filename}`.",
            f"- Forecast target: `{target_column}`.",
            "",
            "## Reduced Feature Sets",
            f"- Candidate A features: `{len(feature_selection['candidate_a_features'])}`.",
            f"- Candidate B features: `{len(feature_selection['candidate_b_features'])}`.",
            f"- Candidate C features: `{len(feature_selection['candidate_c_features'])}`.",
            "",
            "## Raw Target Statistics",
            f"- Mean: `{target_raw['mean']:.3f}`.",
            f"- Standard deviation: `{target_raw['std']:.3f}`.",
            f"- Minimum: `{target_raw['min']:.3f}`.",
            f"- Maximum: `{target_raw['max']:.3f}`.",
            "",
            "## Data Quality",
            quality_report.strip(),
        ]
    )


def build_target_profiling_markdown(
    target_column: str,
    granularity_summary: pd.DataFrame,
    autocorrelation_summary: pd.DataFrame,
    stationarity_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
) -> str:
    strongest_correlations = (
        correlation_summary.sort_values(["granularity", "abs_correlation"], ascending=[True, False])
        .groupby("granularity", as_index=False)
        .head(5)
    )
    return "\n".join(
        [
            "# Target Profiling Report",
            "",
            "## Goal",
            f"- Profile target `{target_column}` in terms of granularity, autocorrelation, and stationarity.",
            "- Summarize which non-target variables move most strongly with the target.",
            "",
            "## Granularity Summary",
            dataframe_to_markdown(granularity_summary, ["id_key", "granularity", "rows", "start_date", "end_date"]),
            "",
            "## Autocorrelation Summary",
            dataframe_to_markdown(
                autocorrelation_summary,
                ["granularity", "frequency", "acf_30s", "acf_1min", "acf_5min", "acf_10min"],
            ),
            "",
            "## Stationarity Summary",
            dataframe_to_markdown(
                stationarity_summary,
                [
                    "granularity",
                    "transform",
                    "adf_pvalue",
                    "kpss_pvalue",
                    "stationary_recommended",
                ],
            ),
            "",
            "## Strongest Target Correlations",
            dataframe_to_markdown(
                strongest_correlations,
                ["granularity", "variable", "correlation", "abs_correlation"],
            ),
        ]
    )


def build_data_preparation_markdown(
    quality_summary: dict[str, Any],
    resampling_policy: dict[str, Any],
    scaling_method: str,
    granularity_summary: pd.DataFrame,
    smoothing_summary: pd.DataFrame,
    stationarity_summary: pd.DataFrame,
) -> str:
    aggregation_rows = granularity_summary.copy()
    smoothing_rows = smoothing_summary[smoothing_summary["transform"] == "trailing_rolling_mean"].copy()
    differencing_rows = stationarity_summary[stationarity_summary["transform"] != "original"].copy()
    return "\n".join(
        [
            "# Data Preparation Report",
            "",
            "## Preparation Sequence",
            "- Missing-value repair.",
            "- Train-fitted target scaling.",
            "- Aggregation to the modeled granularities.",
            "- First- and second-difference candidate generation.",
            "- Train-only smoothing for MLP candidate selection.",
            "",
            "## Missing-Value Repair",
            f"- Missing cells before repair: `{int(quality_summary['missing_cells'])}`.",
            f"- Cells repaired from `data_AutoReg.csv`: `{int(quality_summary['changed_cells'])}`.",
            "- The repaired frame becomes the only source for later transformations and forecasts.",
            "",
            "## Scaling",
            f"- Active scaling method: `{scaling_method}`.",
            "- Scaling is fitted on train only so validation and test remain unseen during preprocessing.",
            "- A linear scaler is used so fitted forecasts can be converted back to the original target scale.",
            "",
            "## Aggregation Levels",
            dataframe_to_markdown(aggregation_rows, ["granularity", "rows", "start_date", "end_date"]),
            "",
            "## Resampling Policy",
            flatten_config_to_markdown_table(resampling_policy),
            "",
            "## Smoothing Diagnostics",
            dataframe_to_markdown(smoothing_rows, ["granularity", "window", "window_steps", "std"]),
            "",
            "## Differencing Diagnostics",
            dataframe_to_markdown(
                differencing_rows,
                ["granularity", "transform", "adf_pvalue", "kpss_pvalue", "stationary_recommended"],
            ),
        ]
    )


def build_forecasting_model_catalog(forecasting_results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    univariate_results = forecasting_results.get("univariate", {})
    if not univariate_results.get("metrics", pd.DataFrame()).empty:
        rows.append(
            {
                "model_key": "univariate_arima",
                "model_label": "Univariate ARIMA",
                "status": "evaluated",
                "granularity_count": int(univariate_results["metrics"]["granularity"].nunique()),
                "report_path": "models/univariate_arima.md",
            }
        )
    if not univariate_results.get("mlp_metrics", pd.DataFrame()).empty:
        rows.append(
            {
                "model_key": "univariate_mlp_target_lags",
                "model_label": "Univariate MLP Target Lags",
                "status": "evaluated",
                "granularity_count": int(univariate_results["mlp_metrics"]["granularity"].nunique()),
                "report_path": "models/univariate_mlp_target_lags.md",
            }
        )
    return pd.DataFrame(rows)


def build_forecasting_comparison_frame(
    forecasting_results: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    univariate_results = forecasting_results.get("univariate", {})
    for row in univariate_results.get("metrics", pd.DataFrame()).to_dict("records"):
        rows.append(
            {
                "model_key": "univariate_arima",
                "model_label": "Univariate ARIMA",
                "split": "test",
                "granularity": row["granularity"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mape": row["mape"],
                "smape": row["smape"],
                "bias": row["bias"],
                "configuration": f"ARIMA({row['p']},{row['d']},{row['q']})",
            }
        )
    for row in univariate_results.get("mlp_metrics", pd.DataFrame()).to_dict("records"):
        rows.append(
            {
                "model_key": "univariate_mlp_target_lags",
                "model_label": "Univariate MLP Target Lags",
                "split": row["split"],
                "granularity": row["granularity"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mape": row["mape"],
                "smape": row["smape"],
                "bias": row["bias"],
                "configuration": (
                    f"{row['transform_name']} | smooth={row['training_smoothing_window']} | "
                    f"lr={row['learning_rate_init']}"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["mae_rank"] = frame.groupby(["split", "granularity"])["mae"].rank(method="dense")
    return frame.sort_values(["split", "granularity", "mae", "rmse"]).reset_index(drop=True)


def collect_forecasting_model_specs(
    target_column: str,
    forecasting_policy: dict[str, Any],
    forecasting_results: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    protocol = forecasting_policy["experimental_protocol"]
    specs = []
    univariate_results = forecasting_results.get("univariate", {})

    arima_metrics = univariate_results.get("metrics", pd.DataFrame())
    if not arima_metrics.empty:
        specs.append(
            {
                "model_key": "univariate_arima",
                "model_label": "Univariate ARIMA",
                "target_column": target_column,
                "protocol": protocol,
                "config": forecasting_policy.get("univariate", {}).get("arima", forecasting_policy.get("arima", {})),
                "metrics": arima_metrics,
                "sections": [],
            }
        )

    mlp_metrics = univariate_results.get("mlp_metrics", pd.DataFrame())
    if not mlp_metrics.empty:
        specs.append(
            {
                "model_key": "univariate_mlp_target_lags",
                "model_label": "Univariate MLP Target Lags",
                "target_column": target_column,
                "protocol": protocol,
                "config": forecasting_policy.get("univariate", {}).get("mlp", {}),
                "metrics": mlp_metrics,
                "sections": [
                    {
                        "title": "Preparation Candidate Ranking",
                        "body": dataframe_to_markdown(
                            univariate_results.get("mlp_preparation_selection", pd.DataFrame()),
                            [
                                "granularity",
                                "candidate_label",
                                "selection_mode",
                                "difference_order",
                                "training_smoothing_window",
                                "learning_rate_init",
                                "validation_mae",
                                "test_mae",
                            ],
                        ),
                    },
                    {
                        "title": "Parameter Effects",
                        "body": dataframe_to_markdown(
                            univariate_results.get("mlp_parameter_effects", pd.DataFrame()),
                            [
                                "granularity",
                                "candidate_label",
                                "hidden_units",
                                "learning_rate_init",
                                "validation_mae",
                                "validation_rmse",
                                "test_mae",
                                "test_rmse",
                            ],
                        ),
                    },
                    {
                        "title": "Prepared Windows",
                        "body": dataframe_to_markdown(
                            univariate_results.get("mlp_window_summary", pd.DataFrame()),
                            [
                                "granularity",
                                "candidate_label",
                                "transform_name",
                                "training_smoothing_window",
                                "train_windows",
                                "validation_windows",
                                "test_windows",
                            ],
                        ),
                    },
                ],
            }
        )
    return specs


def build_forecasting_overview_markdown(
    target_column: str,
    forecasting_policy: dict[str, Any],
    catalog_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
) -> str:
    protocol = forecasting_policy["experimental_protocol"]
    best_rows = summarize_best_results(comparison_frame)
    return "\n".join(
        [
            "# Forecasting Evaluation",
            "",
            "## Scope",
            f"- Target variable: `{target_column}`.",
            f"- Modeled granularities: `{', '.join(protocol['target_granularities'])}`.",
            f"- Forecast horizon: `{protocol['forecast_horizon']}`.",
            f"- Lookback window: `{protocol['lookback_window']}`.",
            "",
            "## Implemented Models",
            dataframe_to_markdown(catalog_frame, ["model_label", "status", "granularity_count", "report_path"]),
            "",
            "## Reported Results By Split And Granularity",
            dataframe_to_markdown(best_rows, ["split", "granularity", "model_label", "mae", "rmse", "configuration"]),
        ]
    )


def build_model_comparison_markdown(comparison_frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Model Comparison",
            "",
            "## Comparable Metrics",
            dataframe_to_markdown(
                comparison_frame,
                [
                    "split",
                    "granularity",
                    "mae_rank",
                    "model_label",
                    "mae",
                    "rmse",
                    "mape",
                    "smape",
                    "bias",
                    "configuration",
                ],
            ),
        ]
    )


def build_model_report_markdown(spec: dict[str, Any]) -> str:
    metrics = spec["metrics"]
    metric_columns = ["granularity", "mae", "rmse", "mape", "smape", "bias"]
    if "split" in metrics.columns:
        metric_columns = [
            "split",
            "granularity",
            "mae",
            "rmse",
            "mape",
            "smape",
            "bias",
            "selection_basis",
            "transform_name",
            "training_smoothing_window",
            "learning_rate_init",
        ]
    if "p" in metrics.columns:
        metric_columns = ["granularity", "p", "d", "q", "mae", "rmse", "mape", "smape", "bias"]

    lines = [
        f"# {spec['model_label']}",
        "",
        "## Scope",
        f"- Target variable: `{spec['target_column']}`.",
        f"- Forecast horizon: `{spec['protocol']['forecast_horizon']}`.",
        f"- Lookback window: `{spec['protocol']['lookback_window']}`.",
        "",
        "## Active Configuration",
        flatten_config_to_markdown_table(spec["config"]),
        "",
        "## Metrics",
        dataframe_to_markdown(metrics, metric_columns),
    ]
    for section in spec["sections"]:
        lines.extend(["", f"## {section['title']}", section["body"]])
    return "\n".join(lines)


def summarize_best_results(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    if comparison_frame.empty:
        return pd.DataFrame(columns=["split", "granularity", "model_label", "mae", "rmse", "configuration"])
    ordered = comparison_frame.sort_values(["split", "granularity", "mae", "rmse", "model_label"])
    return ordered.groupby(["split", "granularity"], as_index=False).first()[
        ["split", "granularity", "model_label", "mae", "rmse", "configuration"]
    ]


def dataframe_to_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return "_No matching columns available._"
    lines = [
        "| " + " | ".join(title_case_column(column) for column in available_columns) + " |",
        "|" + "|".join(["---"] * len(available_columns)) + "|",
    ]
    for row in frame[available_columns].to_dict("records"):
        values = [format_markdown_value(row[column]) for column in available_columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def flatten_config_to_markdown_table(config: dict[str, Any]) -> str:
    rows = []
    for key, value in flatten_config_dict("", config):
        rows.append({"key": key, "value": value})
    if not rows:
        return "_No configuration values available._"
    return dataframe_to_markdown(pd.DataFrame(rows), ["key", "value"])


def flatten_config_dict(prefix: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_config_dict(next_prefix, item))
        return rows
    if isinstance(value, list):
        return [(prefix, json.dumps(value))]
    return [(prefix, str(value))]


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("\n", " ").replace("|", "/")


def title_case_column(column: str) -> str:
    return column.replace("_", " ").title()
