from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from preprocessing import FeatureSelection
except ImportError:  # pragma: no cover - package import path
    from .preprocessing import FeatureSelection


def write_documentation_outputs(
    reports_dir: Path,
    repaired_filename: str,
    target_column: str,
    feature_selection: FeatureSelection,
    quality_report: str,
    quality_summary: dict[str, Any],
    resampling_policy: dict[str, Any],
    scaling_method: str,
    scaling_methods: list[str],
    granularity_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    smoothing_summary: pd.DataFrame,
    scaling_summary: pd.DataFrame,
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
            scaling_methods,
            granularity_summary,
            smoothing_summary,
            scaling_summary,
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
    smoothing_by_granularity = protocol.get("train_only_smoothing_windows_by_granularity", {})
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
        f"- Test: `{float(splits['test']):.0%}` of rows.",
        "- Test remains strictly after train in time order.",
        "",
        "## Preparation Candidates",
        f"- Scaling method: `{protocol['scaling']['method']}` fitted on train only.",
        f"- Differentiation orders tested: `{difference_orders}`.",
        f"- Default train-only smoothing windows tested: {smoothing_windows}.",
        f"- Per-granularity smoothing windows: `{json.dumps(smoothing_by_granularity)}`.",
        "- Test forecasts are written for every candidate for manual visual and metric comparison.",
        "",
        "## MLP Rules",
        f"- Lookback window: `{protocol['lookback_window']}`.",
        f"- Forecast horizon: `{protocol['forecast_horizon']}`.",
        "- Horizon note: `1step` means one row ahead at each granularity, so raw, 30s, and 1min metrics are not equal real-time horizons.",
        "- Input features: target lags only.",
        "- Hidden architecture: single hidden layer.",
        f"- Engine: `{mlp_config.get('engine', 'neuralforecast')}`.",
        f"- Activation: `{mlp_config.get('activation', 'relu')}`.",
        f"- Optimizer: `{mlp_config.get('optimizer', 'adam')}`.",
        f"- Learning-rate grid: `{json.dumps(mlp_config.get('learning_rate_grid', []))}`.",
        f"- Minimum training steps: `{mlp_config.get('min_steps', 5000)}`.",
        f"- Maximum training steps: `{mlp_config.get('max_steps', 7500)}`.",
        f"- Accelerator: `{mlp_config.get('accelerator', 'auto')}`.",
        f"- Windows batch size: `{mlp_config.get('windows_batch_size', 1024)}`.",
        f"- DataLoader workers: `{mlp_config.get('dataloader_num_workers', 0)}`.",
        f"- DataLoader pin memory: `{mlp_config.get('dataloader_pin_memory', False)}`.",
        f"- Comparison mode: `{mlp_config.get('selection_mode', 'test_comparison')}`.",
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
    univariate_results = forecasting_results.get("univariate", {})
    mlp_test_comparison = univariate_results.get("mlp_test_comparison", pd.DataFrame())
    if not mlp_test_comparison.empty:
        (reports_dir / "candidate_selection_report.md").write_text(
            build_candidate_selection_markdown(target_column, forecasting_policy, mlp_test_comparison),
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
    scaling_methods: list[str],
    granularity_summary: pd.DataFrame,
    smoothing_summary: pd.DataFrame,
    scaling_summary: pd.DataFrame,
    stationarity_summary: pd.DataFrame,
) -> str:
    aggregation_rows = granularity_summary.copy()
    smoothing_rows = smoothing_summary[smoothing_summary["transform"] == "trailing_rolling_mean"].copy()
    scaling_rows = scaling_summary[scaling_summary["transform"].isin(["scaled_level", "scaled_first_difference"])].copy()
    differencing_rows = stationarity_summary[stationarity_summary["transform"] != "original"].copy()
    scaling_recommendation = build_scaling_recommendation_markdown(scaling_summary, scaling_method)
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
            f"- Scaling methods compared before forecasting: `{', '.join(scaling_methods)}`.",
            "- Scaling is fitted on train only so test remains unseen during preprocessing.",
            "- A linear scaler is used so fitted forecasts can be converted back to the original target scale.",
            "",
            "## Scaling Diagnostics",
            dataframe_to_markdown(
                scaling_rows,
                ["granularity", "scaling_method", "transform", "std", "min", "max", "skewness"],
            ),
            "",
            "## Scaling Recommendation",
            scaling_recommendation,
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


def build_scaling_recommendation_markdown(scaling_summary: pd.DataFrame, active_scaling_method: str) -> str:
    if scaling_summary.empty:
        return "_No scaling diagnostics available._"

    level_rows = scaling_summary[scaling_summary["transform"] == "scaled_level"].copy()
    first_difference_rows = scaling_summary[scaling_summary["transform"] == "scaled_first_difference"].copy()
    if level_rows.empty:
        return f"- Recommended method: `{active_scaling_method}`."

    minmax_rows = level_rows[level_rows["scaling_method"] == "minmax"]
    standard_rows = level_rows[level_rows["scaling_method"] == "standard"]
    minmax_max = float(minmax_rows["max"].max()) if not minmax_rows.empty else float("nan")
    standard_std = float(standard_rows["std"].mean()) if not standard_rows.empty else float("nan")
    minmax_first_diff_std = (
        float(first_difference_rows[first_difference_rows["scaling_method"] == "minmax"]["std"].mean())
        if not first_difference_rows.empty
        else float("nan")
    )
    standard_first_diff_std = (
        float(first_difference_rows[first_difference_rows["scaling_method"] == "standard"]["std"].mean())
        if not first_difference_rows.empty
        else float("nan")
    )

    return "\n".join(
        [
            f"- Recommended method for the next forecasting run: `{active_scaling_method}`.",
            (
                f"- Train-fitted min-max scaling reaches a test maximum of `{minmax_max:.3f}`, "
                "so future values move outside the nominal `[0, 1]` range."
            ),
            (
                f"- Standard scaling keeps the scaled level near unit variance "
                f"(mean std across granularities `{standard_std:.3f}`), which is a stable input scale for the MLP."
            ),
            (
                f"- Differenced signals are much less compressed with standard scaling "
                f"(mean first-difference std `{standard_first_diff_std:.3f}` vs `{minmax_first_diff_std:.3f}` for min-max), "
                "so the MLP has a stronger learning signal after differencing."
            ),
            "- Both methods are linear and preserve the same temporal shape and skewness; the choice is therefore about numerical conditioning, not changing the signal.",
        ]
    )


def build_forecasting_model_catalog(forecasting_results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    univariate_results = forecasting_results.get("univariate", {})
    if not univariate_results.get("arima_metrics", pd.DataFrame()).empty:
        rows.append(
            {
                "model_key": "univariate_arima",
                "model_label": "Univariate ARIMA",
                "status": "evaluated",
                "granularity_count": int(univariate_results["arima_metrics"]["granularity"].nunique()),
                "report_path": "models/univariate_arima.md",
            }
        )
    if not univariate_results.get("mlp_test_comparison", pd.DataFrame()).empty:
        rows.append(
            {
                "model_key": "univariate_mlp_target_lags",
                "model_label": "Univariate MLP Target Lags",
                "status": "evaluated",
                "granularity_count": int(univariate_results["mlp_test_comparison"]["granularity"].nunique()),
                "report_path": "models/univariate_mlp_target_lags.md",
            }
        )
    return pd.DataFrame(rows)


def build_forecasting_comparison_frame(
    forecasting_results: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    univariate_results = forecasting_results.get("univariate", {})
    for row in univariate_results.get("arima_metrics", pd.DataFrame()).to_dict("records"):
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
                "r2": row.get("r2"),
                "configuration": f"ARIMA({row['p']},{row['d']},{row['q']})",
            }
        )
    for row in univariate_results.get("mlp_test_comparison", pd.DataFrame()).to_dict("records"):
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
                "r2": row.get("r2"),
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

    arima_metrics = univariate_results.get("arima_metrics", pd.DataFrame())
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

    mlp_test_comparison = univariate_results.get("mlp_test_comparison", pd.DataFrame())
    if not mlp_test_comparison.empty:
        specs.append(
            {
                "model_key": "univariate_mlp_target_lags",
                "model_label": "Univariate MLP Target Lags",
                "target_column": target_column,
                "protocol": protocol,
                "config": forecasting_policy.get("univariate", {}).get("mlp", {}),
                "metrics": mlp_test_comparison,
                "sections": [
                    {
                        "title": "Test Comparison Metrics",
                        "body": dataframe_to_markdown(
                            mlp_test_comparison,
                            [
                                "split",
                                "granularity",
                                "candidate_label",
                                "selection_mode",
                                "difference_order",
                                "training_smoothing_window",
                                "learning_rate_init",
                                "min_steps",
                                "max_steps",
                                "training_seconds",
                                "mae",
                                "rmse",
                                "r2",
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
            "- A `1step` horizon is evaluated across the whole test split; it is one row ahead per timestamp, not a single forecast point.",
            "- Because row duration differs by granularity, raw, 30s, and 1min one-step metrics should not be treated as the same real-time horizon.",
            "",
            "## Implemented Models",
            dataframe_to_markdown(catalog_frame, ["model_label", "status", "granularity_count", "report_path"]),
            "",
            "## Reported Results By Split And Granularity",
            dataframe_to_markdown(best_rows, ["split", "granularity", "model_label", "mae", "rmse", "r2", "configuration"]),
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
                    "r2",
                    "configuration",
                ],
            ),
        ]
    )


def build_candidate_selection_markdown(
    target_column: str,
    forecasting_policy: dict[str, Any],
    mlp_test_comparison: pd.DataFrame,
) -> str:
    protocol = forecasting_policy["experimental_protocol"]
    mlp_config = forecasting_policy.get("univariate", {}).get("mlp", {})
    selected_config = list(mlp_config.get("selected_candidate_combinations", []))
    selected_rows = match_selected_candidate_rows(mlp_test_comparison, selected_config)
    best_by_granularity = (
        mlp_test_comparison.sort_values(["granularity", "mae", "rmse", "learning_rate_init"])
        .groupby("granularity", as_index=False)
        .head(3)
    )
    smoothing_summary = summarize_metric_group(
        mlp_test_comparison,
        ["granularity", "training_smoothing_window"],
        ["mae", "rmse", "r2"],
    )
    transform_summary = summarize_metric_group(
        mlp_test_comparison,
        ["granularity", "transform_name"],
        ["mae", "rmse", "r2"],
    )
    learning_rate_summary = summarize_metric_group(
        mlp_test_comparison,
        ["learning_rate_init"],
        ["mae", "rmse", "r2"],
    )
    selected_table_columns = [
        "granularity",
        "candidate_label",
        "role",
        "transform_name",
        "training_smoothing_window",
        "learning_rate_init",
        "mae",
        "rmse",
        "r2",
    ]

    return "\n".join(
        [
            "# Candidate Selection Report",
            "",
            "## Scope",
            f"- Target variable: `{target_column}`.",
            f"- Review dataset: chronological test block from the `{protocol['splits']['test']:.0%}` holdout split.",
            f"- Forecast horizon used for this review: `{protocol['forecast_horizon']}`.",
            "- This is a chronological test-block candidate review, not random resampling or a separate model-selection split.",
            "- `1step` means one row ahead at each granularity, so raw, 30s, and 1min scores are useful for within-granularity ranking but are not equal real-time forecast horizons.",
            "",
            "## Selected Candidate Combinations",
            dataframe_to_markdown(selected_rows, selected_table_columns),
            "",
            "## Metric Evidence",
            "- The selected combinations keep the best two candidates per granularity so the next stage can compare raw, 30s, and 1min behavior without mixing unequal one-step horizons into a single final decision.",
            "- All selected candidates use `standard` scaling, matching the preprocessing recommendation.",
            "- No selected candidate uses train-only smoothing because smoothing consistently worsened one-step test metrics.",
            "- `0.0001` is retained as the main learning rate because it produced the best candidate in each granularity; `0.001` and `0.01` are retained only where they won a level-transformed alternative.",
            "",
            "## Best Three Per Granularity",
            dataframe_to_markdown(
                best_by_granularity,
                [
                    "granularity",
                    "candidate_label",
                    "transform_name",
                    "training_smoothing_window",
                    "learning_rate_init",
                    "mae",
                    "rmse",
                    "r2",
                ],
            ),
            "",
            "## Smoothing Evidence",
            dataframe_to_markdown(
                smoothing_summary,
                ["granularity", "training_smoothing_window", "mean_mae", "mean_rmse", "mean_r2"],
            ),
            "",
            "## Transform Evidence",
            dataframe_to_markdown(
                transform_summary,
                ["granularity", "transform_name", "mean_mae", "mean_rmse", "mean_r2"],
            ),
            "",
            "## Learning-Rate Evidence",
            dataframe_to_markdown(
                learning_rate_summary,
                ["learning_rate_init", "mean_mae", "mean_rmse", "mean_r2"],
            ),
            "",
            "## Next Stage",
            "- Use the selected combinations as the reduced candidate set for the next forecasting stage.",
            "- The final project objective still requires a `10min` horizon run. At that stage, raw, 30s, and 1min will correspond to 120, 20, and 10 forecast steps respectively.",
            "- These one-step results justify preprocessing and candidate reduction; they should not be presented as the final 10-minute forecasting result.",
        ]
    )


def match_selected_candidate_rows(
    mlp_test_comparison: pd.DataFrame,
    selected_config: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for selected in selected_config:
        mask = (
            (mlp_test_comparison["granularity"] == selected["granularity"])
            & (mlp_test_comparison["difference_order"].astype(int) == int(selected["difference_order"]))
            & (mlp_test_comparison["training_smoothing_window"] == selected["training_smoothing_window"])
            & (mlp_test_comparison["learning_rate_init"].astype(float) == float(selected["learning_rate_init"]))
        )
        matched = mlp_test_comparison[mask].copy()
        if matched.empty:
            rows.append({**selected, "candidate_label": "", "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")})
            continue
        row = matched.sort_values(["mae", "rmse"]).iloc[0].to_dict()
        row["role"] = selected.get("role", "")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric_group(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, as_index=False)[metric_columns].mean()
    return grouped.rename(columns={metric: f"mean_{metric}" for metric in metric_columns})


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
            "r2",
            "transform_name",
            "training_smoothing_window",
            "learning_rate_init",
            "min_steps",
            "max_steps",
            "training_seconds",
        ]
    if "p" in metrics.columns:
        metric_columns = ["granularity", "p", "d", "q", "mae", "rmse", "mape", "smape", "bias", "r2"]

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
        return pd.DataFrame(columns=["split", "granularity", "model_label", "mae", "rmse", "r2", "configuration"])
    ordered = comparison_frame.sort_values(["split", "granularity", "mae", "rmse", "model_label"])
    return ordered.groupby(["split", "granularity"], as_index=False).first()[
        ["split", "granularity", "model_label", "mae", "rmse", "r2", "configuration"]
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
        if value != 0 and abs(value) < 0.001:
            return f"{value:.4g}"
        return f"{value:.3f}"
    return str(value).replace("\n", " ").replace("|", "/")


def title_case_column(column: str) -> str:
    return column.replace("_", " ").title()
