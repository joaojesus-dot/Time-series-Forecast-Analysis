from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extraction import DATASET_ANALYZERS, build_boiler_quality_summary
    from exploration import (
        build_granularity_summary,
        build_stationarity_summary,
        build_target_autocorrelation_summary,
        build_target_correlation_summary,
        summarize_target_transforms,
        summarize_variables,
    )
    from plots import (
        write_historical_model_metric_plots,
        write_plot_specs,
        write_target_correlation_barplots,
        write_target_profiling_plots,
    )
    from preprocessing import (
        build_feature_selection,
        build_feature_subset,
        build_granularity_versions,
        build_scaled_target_transform_series,
        build_target_transform_series,
        load_repaired_boiler_frame,
    )
    from reports import (
        build_forecasting_comparison_frame,
        build_forecasting_model_catalog,
        write_documentation_outputs,
        write_experiment_plan,
        write_forecasting_report_suite,
    )
else:
    from .extraction import DATASET_ANALYZERS, build_boiler_quality_summary
    from .exploration import (
        build_granularity_summary,
        build_stationarity_summary,
        build_target_autocorrelation_summary,
        build_target_correlation_summary,
        summarize_target_transforms,
        summarize_variables,
    )
    from .plots import (
        write_historical_model_metric_plots,
        write_plot_specs,
        write_target_correlation_barplots,
        write_target_profiling_plots,
    )
    from .preprocessing import (
        build_feature_selection,
        build_feature_subset,
        build_granularity_versions,
        build_scaled_target_transform_series,
        build_target_transform_series,
        load_repaired_boiler_frame,
    )
    from .reports import (
        build_forecasting_comparison_frame,
        build_forecasting_model_catalog,
        write_documentation_outputs,
        write_experiment_plan,
        write_forecasting_report_suite,
    )


CONFIG_DIR = Path(__file__).resolve().parent / "config"


def log_step(message: str) -> None:
    print(f"[orchestrator] {message}", flush=True)


def write_run_status(path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    path.write_text(f"{timestamp} | {message}\n", encoding="utf-8")


def load_forecasting_pipeline() -> Any:
    if __package__ in {None, ""}:
        from forecasting import run_forecasting_pipeline
    else:
        from .forecasting import run_forecasting_pipeline
    return run_forecasting_pipeline


def migrate_pre_current_active_artifacts(dataset_output: Path) -> None:
    """Move the former flat active artifact tree aside once, keeping history untouched."""
    legacy_root = dataset_output / "legacy" / "pre_current_refactor"
    legacy_root.mkdir(parents=True, exist_ok=True)
    # Only the previous active folders are moved. `history/` is deliberately excluded.
    for name in ["reports", "tables", "plots", "forecasting", "run_status.txt"]:
        source = dataset_output / name
        if not source.exists():
            continue
        destination = legacy_root / name
        if destination.exists():
            # If the migration has already happened once, preserve both snapshots.
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            destination = legacy_root / f"{source.stem}_{timestamp}{source.suffix}" if source.is_file() else legacy_root / f"{name}_{timestamp}"
        shutil.move(str(source), str(destination))


def seed_current_story_artifacts(dataset_output: Path, current_output: Path) -> None:
    """Promote known-good legacy/history evidence into empty current stage folders."""
    legacy_root = dataset_output / "legacy" / "pre_current_refactor"
    stage0_output = current_output / "stage_0_preprocessing"
    if legacy_root.exists():
        # Stage 0 evidence lives in the old flat preprocessing reports/tables/plots.
        if not any(path.is_file() for path in (stage0_output / "reports").rglob("*")):
            _copy_artifact_file_if_exists(legacy_root / "reports" / "dataset_overview.md", stage0_output / "reports" / "dataset_overview.md")
            _copy_artifact_file_if_exists(
                legacy_root / "reports" / "data_preparation_report.md",
                stage0_output / "reports" / "data_preparation_report.md",
            )
            _copy_artifact_file_if_exists(
                legacy_root / "reports" / "target_profiling_report.md",
                stage0_output / "reports" / "target_profiling_report.md",
            )
        if not any(path.is_file() for path in (stage0_output / "tables").rglob("*")):
            _copy_artifact_tree_if_exists(
                legacy_root / "tables" / "pre_forecasting_summary",
                stage0_output / "tables" / "pre_forecasting_summary",
            )
            _copy_artifact_file_if_exists(
                legacy_root / "tables" / "pre_forecasting_summary.xlsx",
                stage0_output / "tables" / "pre_forecasting_summary.xlsx",
            )
        if not any(path.is_file() for path in (stage0_output / "plots").rglob("*")):
            for plot_folder in ["exploratory", "pre_forecasting", "reduced_subsets"]:
                _copy_artifact_tree_if_exists(legacy_root / "plots" / plot_folder, stage0_output / "plots" / plot_folder)

    stage1_output = current_output / "stage_1_screening"
    if not any(path.is_file() for path in stage1_output.rglob("*")):
        # Stage 1 evidence is the archived 72-fit one-step MLP screening run.
        stage1_archive = find_stage1_screening_archive(dataset_output / "history")
        if stage1_archive is not None:
            _copy_artifact_tree_if_exists(stage1_archive / "reports", stage1_output / "reports")
            _copy_artifact_tree_if_exists(stage1_archive / "tables", stage1_output / "tables")
            _copy_artifact_tree_if_exists(stage1_archive / "plots", stage1_output / "plots")


def find_stage1_screening_archive(history_root: Path) -> Path | None:
    """Find the latest archived run that contains the full one-step MLP screening evidence."""
    if not history_root.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for manifest_path in history_root.glob("*/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        policy = manifest.get("forecasting_policy", {})
        active_stage = str(policy.get("active_stage", ""))
        horizon = str(policy.get("experimental_protocol", {}).get("forecast_horizon", ""))
        trained_count = int(str(manifest.get("trained_model_count", "0") or "0"))
        run_tag = str(manifest.get("run_tag", ""))
        if (
            horizon == "1step"
            and trained_count >= 72
            and (active_stage == "all_candidates_1step_review" or "1step-review" in run_tag)
        ):
            candidates.append((str(manifest.get("run_created_at", manifest_path.parent.name)), manifest_path.parent))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _copy_artifact_tree_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _copy_artifact_file_if_exists(source: Path, destination: Path) -> None:
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def resolve_current_stage_output(current_output: Path, forecasting_config: dict[str, Any], skip_forecasting: bool) -> Path:
    """Map the active workflow stage to a human-readable current artifact folder."""
    # Stage 0 does not have a forecasting stage name, so it is keyed by the skip flag.
    if skip_forecasting:
        return current_output / "stage_0_preprocessing"

    stage_name = str(forecasting_config.get("active_stage", "")).strip()
    # Stage 2 baselines get separate folders because they are both current evidence.
    stage_dirs = {
        "stage_1_mlp_screening_1step": current_output / "stage_1_screening",
        "stage_2_baseline_1step_top3": current_output / "stage_2_horizon_impact" / "stage_2_baseline_1step_top3",
        "stage_2_baseline_3min_top3": current_output / "stage_2_horizon_impact" / "stage_2_baseline_3min_top3",
        "stage_3_univariate_mlp_3min": current_output / "stage_3_advanced_models" / "univariate_mlp",
        "stage_3_univariate_arima_3min": current_output / "stage_3_advanced_models" / "univariate_arima",
        "stage_3_univariate_exponential_smoothing_3min": current_output
        / "stage_3_advanced_models"
        / "univariate_exponential_smoothing",
    }
    # Unknown experimental stages are isolated under Stage 3 instead of polluting active baselines.
    return stage_dirs.get(stage_name, current_output / "stage_3_advanced_models" / sanitize_run_tag(stage_name or "forecasting"))


def is_model_specific_stage(forecasting_config: dict[str, Any]) -> bool:
    """Stage 3 folders already encode the model family, so artifacts do not need extra nesting."""
    stage_name = str(forecasting_config.get("active_stage", "")).strip()
    return stage_name.startswith("stage_3_univariate_")


def write_artifact_index(current_output: Path) -> None:
    """Write the small human map for the current output tree."""
    current_output.mkdir(parents=True, exist_ok=True)
    content = """# Current Artifact Index

This folder contains the active, presentation-facing outputs for the industrial boiler temperature forecasting project. Older active artifacts from the previous flat tree were moved to `../legacy/pre_current_refactor/`; archived forecasting runs remain under `../history/`. Stage 0 and Stage 1 currently preserve evidence promoted from the latest reliable pre-refactor artifacts, because those stages were not rerun after the staged output-tree refactor.

## Stage 0 - Preprocessing

Path: `stage_0_preprocessing/`

- `reports/`: data preparation, target profiling, scaling, smoothing, differencing, stationarity, and correlation reports.
- `tables/pre_forecasting_summary/`: compact CSV tables used to support the preprocessing discussion.
- `plots/`: exploratory, reduced-subset, and pre-forecasting target diagnostic plots.
- Current source: promoted from `../legacy/pre_current_refactor/`.

## Stage 1 - MLP Screening At One Step

Path: `stage_1_screening/`

- Purpose: run the assignment-style MLP over all 72 preprocessing candidates at a `1step` horizon.
- `forecasting/`: full forecast CSVs and MLP training history when enabled.
- `reports/forecasting/`: experiment plan, candidate selection, model comparison, and MLP notes.
- `tables/forecasting_summary/`: screening metrics and selected-candidate tables.
- `plots/forecasting/`: ranking plots plus useful training loss and learning-curve diagnostics.
- Current source: promoted from archived run `../history/20260430T014551_standard-real-mlp-1step-review_dc82a37d/`.

## Stage 2 - Horizon Impact Baseline

Path: `stage_2_horizon_impact/`

- `stage_2_baseline_1step_top3/`: baseline MLP on the best candidate per granularity at `1step`.
- `stage_2_baseline_3min_top3/`: the same baseline MLP candidates at the project `3min` horizon.
- `comparison/`: offline comparison plots and CSVs for the same candidates across horizons.

## Stage 3 - Advanced 3-Minute Models

Path: `stage_3_advanced_models/`

- `univariate_mlp/`: first runnable Stage 3 model family. It compares two-hidden-layer NeuralForecast MLP variants at the 3-minute horizon. Because the parent folder already names the model family, its reports, forecast CSVs, and plots are written directly under `univariate_mlp/reports/`, `univariate_mlp/forecasting/`, and `univariate_mlp/plots/`.
- `univariate_arima/`: future ARIMA workspace, blocked until model settings are reviewed.
- `univariate_exponential_smoothing/`: future smoothing-model workspace, blocked until model settings are reviewed.

## Legacy Follow-Up

Review `../legacy/pre_current_refactor/` after this migration. Files that still help the project story should be promoted into one of the staged folders above; files that do not help should be excluded from Git commits or removed intentionally.
"""
    (current_output / "artifact_index.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write boiler-only KDD and forecasting outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-forecasting", action="store_true")
    parser.add_argument(
        "--forecasting-from-derived",
        action="store_true",
        help="Skip preprocessing and load existing data/<dataset>/derived/subset_B_* CSV files for forecasting.",
    )
    parser.add_argument("--forecasting-stage", type=str, default=None)
    parser.add_argument(
        "--forecasting-models",
        type=str,
        default=None,
        help=(
            "Comma-separated univariate model families to run: mlp, arima, ets. "
            "Example: --forecasting-models mlp or --forecasting-models arima,ets."
        ),
    )
    parser.add_argument("--forecasting-smoke-test", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-learning-rates", type=int, default=None)
    args = parser.parse_args()
    if args.skip_forecasting and args.forecasting_from_derived:
        parser.error("--forecasting-from-derived cannot be combined with --skip-forecasting.")
    log_step("Starting orchestrator.")

    dataset_name = "chinese_boiler_dataset"
    dataset_output = args.output_dir / dataset_name
    current_output = dataset_output / "current"
    migrate_pre_current_active_artifacts(dataset_output)
    seed_current_story_artifacts(dataset_output, current_output)
    reports_output = current_output / "stage_0_preprocessing" / "reports"
    tables_output = current_output / "stage_0_preprocessing" / "tables"
    plots_output = current_output / "stage_0_preprocessing" / "plots"
    exploratory_plots_output = plots_output / "exploratory"
    reduced_plots_output = plots_output / "reduced_subsets"
    pre_forecasting_plots_output = plots_output / "pre_forecasting"
    profiling_plots_output = pre_forecasting_plots_output / "target_profiling"
    correlation_plots_output = pre_forecasting_plots_output / "target_correlations"
    derived_data_output = args.data_dir / dataset_name / "derived"

    prepare_output_tree(
        dataset_output,
        args.data_dir / dataset_name,
        preserve_forecasting_outputs=True,
        preserve_derived_data=args.forecasting_from_derived,
    )
    for directory in [
        reports_output,
        tables_output,
        exploratory_plots_output,
        reduced_plots_output,
        pre_forecasting_plots_output,
        profiling_plots_output,
        correlation_plots_output,
        derived_data_output,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    status_path = current_output / "run_status.txt"
    write_run_status(status_path, "started")

    family_reduction_rows = json.loads((CONFIG_DIR / "boiler_family_reduction.json").read_text(encoding="utf-8"))
    preprocessing_config = json.loads((CONFIG_DIR / "boiler_preprocessing.json").read_text(encoding="utf-8"))
    forecasting_config = json.loads((CONFIG_DIR / "boiler_forecasting.json").read_text(encoding="utf-8"))
    if args.forecasting_stage is not None:
        forecasting_config["active_stage"] = args.forecasting_stage
    apply_forecasting_stage_config(forecasting_config)
    apply_forecasting_runtime_overrides(forecasting_config, args)
    stage_output = resolve_current_stage_output(current_output, forecasting_config, args.skip_forecasting)
    forecasting_output = stage_output / "forecasting"
    if is_model_specific_stage(forecasting_config):
        forecasting_reports_output = stage_output / "reports"
        forecasting_plots_output = stage_output / "plots"
    else:
        forecasting_reports_output = stage_output / "reports" / "forecasting"
        forecasting_plots_output = stage_output / "plots" / "forecasting"
    forecasting_tables_output = stage_output / "tables"
    for directory in [
        forecasting_output,
        forecasting_reports_output,
        forecasting_plots_output,
        forecasting_tables_output,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    log_step(
        "Loaded configs: "
        f"stage={forecasting_config.get('active_stage', 'default')} "
        f"horizon={forecasting_config['experimental_protocol']['forecast_horizon']} "
        f"skip_forecasting={args.skip_forecasting} "
        f"forecasting_from_derived={args.forecasting_from_derived}."
    )
    write_run_status(
        status_path,
        f"loaded configs; stage={forecasting_config.get('active_stage', 'default')}; "
        f"horizon={forecasting_config['experimental_protocol']['forecast_horizon']}; "
        f"skip_forecasting={args.skip_forecasting}; "
        f"forecasting_from_derived={args.forecasting_from_derived}",
    )
    history_root = dataset_output / str(forecasting_config.get("run_tracking", {}).get("history_root", "history"))
    write_artifact_index(current_output)
    plot_config = json.loads((CONFIG_DIR / "boiler_plot_specs.json").read_text(encoding="utf-8"))
    feature_selection = build_feature_selection(family_reduction_rows)
    target_column = str(feature_selection["target_column"])
    candidate_b_features = list(feature_selection["candidate_b_features"])
    candidate_c_features = list(feature_selection["candidate_c_features"])
    granularity_options = preprocessing_config["granularity_options"]
    resampling_policy = preprocessing_config["resampling_policy"]
    profiling_granularities = list(forecasting_config["experimental_protocol"].get("profiling_granularities", []))
    scaling_method = str(forecasting_config["experimental_protocol"]["scaling"]["method"])
    scaling_methods = [
        str(method)
        for method in forecasting_config["experimental_protocol"]["scaling"].get(
            "comparison_methods",
            [scaling_method],
        )
    ]
    if scaling_method not in scaling_methods:
        scaling_methods.insert(0, scaling_method)
    train_fraction = float(forecasting_config["experimental_protocol"]["splits"]["train"])

    if args.forecasting_from_derived:
        log_step("Loading derived preprocessing artifacts for forecasting.")
        write_run_status(status_path, "loading derived preprocessing artifacts for forecasting")
        subset_b_datasets = load_derived_forecasting_datasets(
            derived_data_output,
            dataset_prefix=str(forecasting_config["experimental_protocol"]["dataset_prefix"]),
            target_granularities=list(forecasting_config["experimental_protocol"]["target_granularities"]),
            timestamp_column=str(resampling_policy["timestamp_column"]),
        )
        run_forecasting_stage(
            subset_b_datasets=subset_b_datasets,
            target_column=target_column,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
            forecasting_config=forecasting_config,
            forecasting_output=forecasting_output,
            forecasting_reports_output=forecasting_reports_output,
            forecasting_plots_output=forecasting_plots_output,
            derived_data_output=derived_data_output,
            tables_output=forecasting_tables_output,
            history_root=history_root,
            status_path=status_path,
        )
        return 0

    # Load the dataset quality summary and the repaired working frame.
    log_step("Loading quality reports and repaired boiler data.")
    quality_report = DATASET_ANALYZERS[dataset_name](args.data_dir / dataset_name)
    quality_summary = build_boiler_quality_summary(args.data_dir / dataset_name)

    repaired = load_repaired_boiler_frame(args.data_dir / dataset_name)
    repaired_path = derived_data_output / "boiler_repaired.csv"
    write_csv_output(repaired, repaired_path)
    log_step(f"Wrote repaired data: {repaired_path}")
    write_run_status(status_path, f"wrote repaired data: {repaired_path}")

    # Build the reduced feature subsets used by the reports and forecasts.
    log_step("Building reduced feature subsets and granularity versions.")
    subset_b_base = build_feature_subset(repaired, target_column, candidate_b_features)
    subset_c_base = build_feature_subset(repaired, target_column, candidate_c_features)
    write_csv_output(subset_c_base, derived_data_output / "subset_C.csv")

    # Generate the raw and aggregated Candidate B datasets.
    subset_b_datasets = build_granularity_versions(
        subset_b_base,
        granularity_options,
        dataset_name="subset_B",
        timestamp_column=str(resampling_policy["timestamp_column"]),
        resampling_policy=resampling_policy,
    )
    for id_key, dataset in subset_b_datasets.items():
        write_csv_output(dataset, derived_data_output / f"{id_key}.csv")
    write_csv_output(subset_b_datasets["subset_B_raw"], derived_data_output / "subset_B.csv")
    log_step(f"Wrote derived datasets to {derived_data_output}.")
    write_run_status(status_path, f"wrote derived datasets to {derived_data_output}")

    granularity_summary = build_granularity_summary(subset_b_datasets, dataset_name="subset_B")
    distribution_summary = summarize_variables(subset_b_datasets)
    profiling_datasets = {
        id_key: dataset
        for id_key, dataset in subset_b_datasets.items()
        if id_key.rsplit("_", 1)[-1] in profiling_granularities
    }

    # Build the cached target transforms used by the preparation summaries and plots.
    log_step("Building preprocessing summaries and target diagnostics.")
    write_run_status(status_path, "building target transform summaries")
    transform_series = build_target_transform_series(subset_b_datasets, target_column, plot_config["smoothing_windows"])
    smoothing_summary = summarize_target_transforms(transform_series, target_column)
    log_step("Built smoothing and differencing summaries.")
    write_run_status(status_path, "building scaling comparison summaries")
    scaling_series = build_scaled_target_transform_series(
        subset_b_datasets,
        target_column,
        plot_config["smoothing_windows"],
        scaling_methods,
        train_fraction,
        timestamp_column=str(resampling_policy["timestamp_column"]),
    )
    scaling_comparison_summary = summarize_target_transforms(scaling_series, target_column)
    log_step("Built scaling comparison summaries.")
    write_run_status(status_path, "building autocorrelation, stationarity, and correlation summaries")
    autocorrelation_summary = build_target_autocorrelation_summary(
        profiling_datasets,
        target_column,
        granularity_options,
        str(resampling_policy["input_frequency"]),
    )
    stationarity_summary = build_stationarity_summary(profiling_datasets, target_column)
    correlation_summary = build_target_correlation_summary(profiling_datasets, target_column)
    log_step("Built autocorrelation, stationarity, and correlation summaries.")

    write_table_bundle(
        tables_output / "pre_forecasting_summary",
        {
            "granularity": granularity_summary,
            "target_distribution": distribution_summary[distribution_summary["variable"] == target_column],
            "target_transforms": smoothing_summary,
            "target_scaling_comparison": scaling_comparison_summary,
            "target_autocorrelation": autocorrelation_summary,
            "target_stationarity": stationarity_summary,
            "target_correlations": correlation_summary,
        },
    )
    log_step(f"Wrote pre-forecasting summary tables to {tables_output / 'pre_forecasting_summary'}.")
    write_run_status(status_path, "writing preprocessing plots")

    frames = {
        "repaired": repaired,
        "subset_C": subset_c_base,
        "smoothing_summary": smoothing_summary,
        "scaling_summary": scaling_comparison_summary,
        **subset_b_datasets,
    }
    output_dirs = {
        "exploratory_plots": exploratory_plots_output,
        "reduced_plots": reduced_plots_output,
        "pre_forecasting_plots": pre_forecasting_plots_output,
    }
    write_plot_specs(
        build_boiler_plot_specs(
            plot_config,
            target_column,
            subset_b_datasets,
            transform_series,
            scaling_series,
            output_dirs,
        ),
        frames,
    )
    write_target_profiling_plots(
        profiling_datasets,
        target_column,
        profiling_plots_output,
        timestamp_column=str(resampling_policy["timestamp_column"]),
    )
    write_target_correlation_barplots(correlation_summary, correlation_plots_output)
    log_step("Wrote preprocessing plots.")
    write_run_status(status_path, "writing documentation reports")

    write_documentation_outputs(
        reports_dir=reports_output,
        repaired_filename=repaired_path.as_posix(),
        target_column=target_column,
        feature_selection=feature_selection,
        quality_report=quality_report,
        quality_summary=quality_summary,
        resampling_policy=resampling_policy,
        scaling_method=scaling_method,
        scaling_methods=scaling_methods,
        granularity_summary=granularity_summary,
        distribution_summary=distribution_summary,
        smoothing_summary=smoothing_summary,
        scaling_summary=scaling_comparison_summary,
        autocorrelation_summary=autocorrelation_summary,
        stationarity_summary=stationarity_summary,
        correlation_summary=correlation_summary,
    )
    log_step("Wrote documentation reports.")
    if not args.skip_forecasting:
        run_forecasting_stage(
            subset_b_datasets=subset_b_datasets,
            target_column=target_column,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
            forecasting_config=forecasting_config,
            forecasting_output=forecasting_output,
            forecasting_reports_output=forecasting_reports_output,
            forecasting_plots_output=forecasting_plots_output,
            derived_data_output=derived_data_output,
            tables_output=forecasting_tables_output,
            history_root=history_root,
            status_path=status_path,
        )
    else:
        log_step("Skipping forecasting stage.")
        write_run_status(status_path, "finished preprocessing only; skipped forecasting")
    return 0


def apply_forecasting_runtime_overrides(forecasting_config: dict[str, Any], args: argparse.Namespace) -> None:
    univariate_config = forecasting_config.setdefault("univariate", {})
    mlp_config = univariate_config.setdefault("mlp", {})
    if args.forecasting_smoke_test:
        mlp_config["candidate_limit"] = 1
        mlp_config["learning_rate_limit"] = 1
        mlp_config["min_steps"] = 1
        mlp_config["max_steps"] = 1
        mlp_config["candidate_settings"] = {}
        forecasting_config.setdefault("run_tracking", {})["tag"] = "smoke-test"
    if args.max_candidates is not None:
        mlp_config["candidate_limit"] = args.max_candidates
    if args.max_learning_rates is not None:
        mlp_config["learning_rate_limit"] = args.max_learning_rates
    if args.forecasting_models is not None:
        apply_forecasting_model_filter(univariate_config, args.forecasting_models)


def apply_forecasting_model_filter(univariate_config: dict[str, Any], configured_models: str) -> None:
    aliases = {
        "mlp": "mlp",
        "arima": "arima",
        "ets": "exponential_smoothing",
        "exponential_smoothing": "exponential_smoothing",
        "exponential-smoothing": "exponential_smoothing",
    }
    requested = {
        aliases.get(item.strip().lower(), item.strip().lower())
        for item in configured_models.split(",")
        if item.strip()
    }
    if not requested:
        raise ValueError("--forecasting-models must include at least one model family.")
    if "all" in requested:
        return
    valid = {"mlp", "arima", "exponential_smoothing"}
    unknown = requested - valid
    if unknown:
        valid_labels = "mlp, arima, ets"
        raise ValueError(f"Unknown --forecasting-models value(s): {', '.join(sorted(unknown))}. Use: {valid_labels}.")

    univariate_config.setdefault("mlp", {})["enabled"] = "mlp" in requested
    univariate_config.setdefault("arima", {})["enabled"] = "arima" in requested
    univariate_config.setdefault("exponential_smoothing", {})["enabled"] = "exponential_smoothing" in requested


def apply_forecasting_stage_config(forecasting_config: dict[str, Any]) -> None:
    stage_name = forecasting_config.get("active_stage")
    if not stage_name:
        return

    stages = forecasting_config.get("stages", {})
    if not isinstance(stages, dict):
        raise TypeError("forecasting stages must be configured as an object.")
    stage_config = stages.get(str(stage_name))
    if not isinstance(stage_config, dict):
        raise ValueError(f"Active forecasting stage '{stage_name}' is not defined in config.")
    if stage_config.get("runnable") is False:
        raise ValueError(
            f"Forecasting stage '{stage_name}' is a placeholder pending review and is not runnable yet."
        )

    protocol = forecasting_config.setdefault("experimental_protocol", {})
    univariate_config = forecasting_config.setdefault("univariate", {})
    mlp_config = univariate_config.setdefault("mlp", {})
    run_tracking = forecasting_config.setdefault("run_tracking", {})

    if "forecast_horizon" in stage_config:
        protocol["forecast_horizon"] = stage_config["forecast_horizon"]
    if "selection_mode" in stage_config:
        mlp_config["selection_mode"] = stage_config["selection_mode"]
    if "use_selected_candidate_combinations" in stage_config:
        mlp_config["use_selected_candidate_combinations"] = bool(stage_config["use_selected_candidate_combinations"])
    if "candidate_selection" in stage_config:
        selection_name = str(stage_config["candidate_selection"])
        candidate_selections = forecasting_config.get("candidate_selections", {})
        if not isinstance(candidate_selections, dict):
            raise TypeError("forecasting candidate_selections must be configured as an object.")
        selected_candidates = candidate_selections.get(selection_name)
        if not isinstance(selected_candidates, list) or not selected_candidates:
            raise ValueError(f"Forecasting candidate selection '{selection_name}' is not defined or is empty.")
        mlp_config["candidate_selection"] = selection_name
        mlp_config["selected_candidate_combinations"] = selected_candidates
    if "run_tracking_tag" in stage_config:
        run_tracking["tag"] = stage_config["run_tracking_tag"]
    mlp_settings = stage_config.get("mlp_settings")
    arima_settings = stage_config.get("arima_settings")
    smoothing_settings = stage_config.get("exponential_smoothing_settings")
    if isinstance(mlp_settings, dict):
        mlp_config.update(mlp_settings)
    if isinstance(arima_settings, dict):
        univariate_config.setdefault("arima", {}).update(arima_settings)
    if isinstance(smoothing_settings, dict):
        univariate_config.setdefault("exponential_smoothing", {}).update(
            smoothing_settings
        )


def load_derived_forecasting_datasets(
    derived_data_dir: Path,
    dataset_prefix: str,
    target_granularities: list[str],
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    datasets = {}
    missing_paths = []
    for granularity in target_granularities:
        id_key = f"{dataset_prefix}_{granularity}"
        path = derived_data_dir / f"{id_key}.csv"
        if not path.exists():
            missing_paths.append(path)
            continue
        frame = pd.read_csv(path)
        if timestamp_column not in frame.columns:
            raise ValueError(f"Derived dataset '{path}' is missing timestamp column '{timestamp_column}'.")
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
        if frame[timestamp_column].isna().any():
            raise ValueError(f"Derived dataset '{path}' has invalid timestamps in '{timestamp_column}'.")
        datasets[id_key] = frame.sort_values(timestamp_column).reset_index(drop=True)

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing derived forecasting datasets. Run Stage 0 preprocessing first, then retry "
            f"--forecasting-from-derived. Missing: {missing}"
        )
    return datasets


def run_forecasting_stage(
    subset_b_datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_config: dict[str, Any],
    forecasting_output: Path,
    forecasting_reports_output: Path,
    forecasting_plots_output: Path,
    derived_data_output: Path,
    tables_output: Path,
    history_root: Path,
    status_path: Path,
) -> None:
    try:
        log_step("Starting forecasting stage.")
        write_run_status(status_path, "starting forecasting stage")
        prune_active_forecasting_run_outputs(
            forecasting_output=forecasting_output,
            forecasting_reports_output=forecasting_reports_output,
            forecasting_plots_output=forecasting_plots_output,
        )
        write_experiment_plan(
            output_path=forecasting_reports_output / "experiment_plan.md",
            target_column=target_column,
            forecasting_policy=forecasting_config,
            resampling_policy=resampling_policy,
        )
        run_metadata = build_run_metadata(forecasting_config)
        run_metadata["started_at"] = datetime.now().isoformat(timespec="seconds")
        run_started = perf_counter()
        log_step("Importing forecasting backend.")
        write_run_status(status_path, "importing forecasting backend")
        run_forecasting_pipeline = load_forecasting_pipeline()
        log_step("Running forecasting pipeline.")
        write_run_status(status_path, "running forecasting pipeline")
        forecasting_results = run_forecasting_pipeline(
            datasets=subset_b_datasets,
            target_column=target_column,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
            forecasting_policy=forecasting_config,
            output_dir=forecasting_output,
            reports_dir=forecasting_reports_output,
            plots_dir=forecasting_plots_output,
            derived_data_dir=derived_data_output,
            timestamp_column=str(resampling_policy["timestamp_column"]),
        )
        forecasting_results["comparison"] = {
            "model_comparison": build_forecasting_comparison_frame(forecasting_results),
            "model_catalog": build_forecasting_model_catalog(forecasting_results),
        }
        run_metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run_metadata["duration_seconds"] = f"{perf_counter() - run_started:.3f}"
        run_metadata["trained_model_count"] = str(
            len(forecasting_results.get("univariate", {}).get("mlp_test_comparison", pd.DataFrame()))
        )
        write_forecasting_summary_tables(tables_output / "forecasting_summary", forecasting_results)
        write_forecasting_report_suite(
            reports_dir=forecasting_reports_output,
            target_column=target_column,
            forecasting_policy=forecasting_config,
            forecasting_results=forecasting_results,
        )
        archive_forecasting_run(
            history_root=history_root,
            run_metadata=run_metadata,
            forecasting_policy=forecasting_config,
            forecasting_output=forecasting_output,
            reports_dir=forecasting_reports_output,
            tables_dir=tables_output / "forecasting_summary",
            tables_workbook=tables_output / "forecasting_summary.xlsx",
            plots_dir=forecasting_plots_output,
            comparison_frame=forecasting_results["comparison"]["model_comparison"],
            catalog_frame=forecasting_results["comparison"]["model_catalog"],
        )
        log_step("Finished forecasting stage.")
        write_run_status(status_path, "finished forecasting stage")
    except Exception as exc:
        write_run_status(status_path, f"failed forecasting stage: {type(exc).__name__}: {exc}")
        raise


def prune_active_forecasting_run_outputs(
    forecasting_output: Path,
    forecasting_reports_output: Path,
    forecasting_plots_output: Path,
) -> None:
    """Clear run-scoped forecasting artifacts before generating a new archive snapshot."""
    dataset_output = forecasting_output.parent.resolve()
    for path in [forecasting_output, forecasting_reports_output, forecasting_plots_output]:
        resolved_path = path.resolve()
        if not (resolved_path == dataset_output or dataset_output in resolved_path.parents):
            raise ValueError(f"Refusing to remove forecasting artifact outside dataset output: {resolved_path}")
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        path.mkdir(parents=True, exist_ok=True)


def prepare_output_tree(
    dataset_output: Path,
    dataset_data_dir: Path,
    preserve_forecasting_outputs: bool = False,
    preserve_derived_data: bool = False,
) -> None:
    """Remove obsolete generated folders from the previous output layout."""
    obsolete_paths = [
        dataset_output / "data",
        dataset_output / "data_quality.md",
        dataset_output / "distribution_analysis.md",
        dataset_output / "exploration_findings.md",
        dataset_output / "feature_selection_note.json",
        dataset_output / "granularity_analysis.md",
        dataset_output / "outlier_analysis.md",
        dataset_output / "physical_analysis.md",
        dataset_output / "presentation_findings.md",
        dataset_output / "smoothing_differencing_analysis.md",
        dataset_output / "plots" / "distribution",
        dataset_output / "plots" / "outliers",
        dataset_output / "plots" / "preprocessing",
        dataset_data_dir / "feature_selection_note.json",
    ]
    if not preserve_derived_data:
        obsolete_paths.append(dataset_data_dir / "derived")
    if not preserve_forecasting_outputs:
        obsolete_paths.extend(
            [
                dataset_output / "forecasting",
                dataset_output / "plots",
                dataset_output / "reports",
                dataset_output / "tables",
            ]
        )
    for path in obsolete_paths:
        remove_generated_path(path, dataset_output, dataset_data_dir)


def remove_generated_path(path: Path, dataset_output: Path, dataset_data_dir: Path) -> None:
    resolved_path = path.resolve()
    allowed_roots = [dataset_output.resolve(), dataset_data_dir.resolve()]
    if not any(resolved_path == root or root in resolved_path.parents for root in allowed_roots):
        raise ValueError(f"Refusing to remove path outside generated output roots: {resolved_path}")
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_csv_output(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def write_table_bundle(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Write small tables as an Excel workbook plus CSV copies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_excel_workbook(output_dir.with_suffix(".xlsx"), tables)
    for table_name, frame in tables.items():
        write_csv_output(frame, output_dir / f"{table_name}.csv")


def write_excel_workbook(output_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.xlsx")
    with pd.ExcelWriter(temp_path) as writer:
        for sheet_name, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    temp_path.replace(output_path)


def write_forecasting_summary_tables(
    output_dir: Path,
    forecasting_results: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Write the compact forecasting summary tables used in the reports."""
    prune_stale_forecasting_summary_tables(output_dir)
    tables = {}
    for analysis_name, result_frames in forecasting_results.items():
        for frame_name, frame in result_frames.items():
            if frame_name in {
                "forecasts",
                "arima_forecasts",
                "exponential_smoothing_forecasts",
                "mlp_forecasts",
                "mlp_training_history",
                "scaled_target",
                "mlp_metrics",
                "mlp_parameter_effects",
            }:
                continue
            if frame.empty:
                continue
            table_name = f"{analysis_name}_{frame_name}"
            tables[table_name] = frame
    if tables:
        write_table_bundle(output_dir, tables)


def prune_stale_forecasting_summary_tables(output_dir: Path) -> None:
    stale_files = [
        "univariate_mlp_metrics.csv",
        "univariate_mlp_parameter_effects.csv",
        "univariate_mlp_preparation_selection.csv",
        "univariate_metrics.csv",
    ]
    for filename in stale_files:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def build_boiler_plot_specs(
    plot_config: dict[str, Any],
    target_column: str,
    subset_b_datasets: dict[str, Any],
    transform_series: list[dict[str, Any]],
    scaling_series: list[dict[str, Any]],
    output_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    # Resolve the dataset keys, target column, transforms, and output paths used by each plot spec.
    dataset_keys = list(subset_b_datasets)
    return [
        _resolve_plot_spec(spec, plot_config, target_column, dataset_keys, transform_series, scaling_series, output_dirs)
        for spec in plot_config["plot_specs"]
    ]


def _resolve_plot_spec(
    spec: dict[str, Any],
    plot_config: dict[str, Any],
    target_column: str,
    dataset_keys: list[str],
    transform_series: list[dict[str, Any]],
    scaling_series: list[dict[str, Any]],
    output_dirs: dict[str, Path],
) -> dict[str, Any]:
    resolved = dict(spec)

    if resolved.pop("groups_ref", None) == "candidate_b_plot_groups":
        resolved["groups"] = plot_config["candidate_b_plot_groups"]
    if resolved.pop("units_ref", None) == "assumed_plot_units":
        resolved["units"] = plot_config["assumed_plot_units"]
    if resolved.pop("dataset_keys_ref", None) == "subset_b_datasets":
        resolved["dataset_keys"] = dataset_keys
    if resolved.pop("target_column_ref", False):
        resolved["target_column"] = target_column
    if resolved.pop("transform_series_ref", False):
        resolved["transform_series"] = transform_series
    if resolved.pop("scaling_series_ref", False):
        resolved["scaling_series"] = scaling_series

    output_dir_ref = resolved.pop("output_dir_ref", None)
    if output_dir_ref:
        resolved["output_dir"] = output_dirs[str(output_dir_ref)]

    output_path_ref = resolved.pop("output_path_ref", None)
    if output_path_ref:
        directory_name, filename = output_path_ref
        resolved["output_path"] = output_dirs[str(directory_name)] / str(filename)

    return resolved


def build_run_metadata(forecasting_config: dict[str, Any]) -> dict[str, str]:
    run_tracking = forecasting_config.get("run_tracking", {})
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    config_fingerprint = hashlib.sha1(
        json.dumps(forecasting_config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:8]
    tag = sanitize_run_tag(str(run_tracking.get("tag", "")).strip())
    parts = [timestamp]
    if tag:
        parts.append(tag)
    parts.append(config_fingerprint)
    run_id = "_".join(parts)
    run_label = f"{timestamp} | {tag or 'untagged'} | {config_fingerprint}"
    return {
        "run_id": run_id,
        "run_label": run_label,
        "run_tag": tag or "untagged",
        "run_created_at": timestamp,
        "config_fingerprint": config_fingerprint,
    }


def sanitize_run_tag(tag: str) -> str:
    if not tag:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-_")
    return normalized.lower()


def archive_forecasting_run(
    history_root: Path,
    run_metadata: dict[str, str],
    forecasting_policy: dict[str, Any],
    forecasting_output: Path,
    reports_dir: Path,
    tables_dir: Path,
    tables_workbook: Path,
    plots_dir: Path,
    comparison_frame: pd.DataFrame,
    catalog_frame: pd.DataFrame,
) -> None:
    run_tracking = forecasting_policy.get("run_tracking", {})
    if not bool(run_tracking.get("enabled", True)):
        return

    run_root = history_root / run_metadata["run_id"]
    if run_root.exists():
        shutil.rmtree(run_root)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    if bool(run_tracking.get("copy_plots", True)):
        (run_root / "plots").mkdir(parents=True, exist_ok=True)

    if forecasting_output.exists():
        shutil.copytree(forecasting_output, run_root / "forecasting", dirs_exist_ok=True)
    shutil.copytree(reports_dir, run_root / "reports" / "forecasting", dirs_exist_ok=True)
    shutil.copytree(tables_dir, run_root / "tables" / "forecasting_summary", dirs_exist_ok=True)
    if tables_workbook.exists():
        shutil.copy2(tables_workbook, run_root / "tables" / tables_workbook.name)
    if bool(run_tracking.get("copy_plots", True)) and plots_dir.exists():
        shutil.copytree(plots_dir, run_root / "plots" / "forecasting", dirs_exist_ok=True)

    manifest = {
        **run_metadata,
        "forecasting_policy": forecasting_policy,
        "forecasting_output_path": str((run_root / "forecasting").as_posix()),
        "reports_path": str((run_root / "reports" / "forecasting").as_posix()),
        "tables_path": str((run_root / "tables" / "forecasting_summary").as_posix()),
        "plots_path": str((run_root / "plots" / "forecasting").as_posix()),
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history_tables_dir = history_root / "_comparison"
    history_tables_dir.mkdir(parents=True, exist_ok=True)
    write_run_history_tables(history_tables_dir, run_metadata, comparison_frame, catalog_frame)
    write_historical_model_metric_plots(
        read_history_table(history_tables_dir / "model_runs.csv"),
        history_root / "_plots",
    )


def write_run_history_tables(
    history_tables_dir: Path,
    run_metadata: dict[str, str],
    comparison_frame: pd.DataFrame,
    catalog_frame: pd.DataFrame,
) -> None:
    comparison_with_run = comparison_frame.copy()
    for key, value in run_metadata.items():
        comparison_with_run[key] = value

    catalog_with_run = catalog_frame.copy()
    for key, value in run_metadata.items():
        catalog_with_run[key] = value

    upsert_history_table(
        history_tables_dir / "model_runs.csv",
        comparison_with_run,
        key_columns=["run_id", "model_key", "split", "granularity", "configuration"],
    )
    upsert_history_table(
        history_tables_dir / "run_catalog.csv",
        catalog_with_run,
        key_columns=["run_id", "model_key"],
    )


def read_history_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def upsert_history_table(path: Path, frame: pd.DataFrame, key_columns: list[str]) -> None:
    if frame.empty:
        return
    existing = read_history_table(path)
    combined = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame.copy()
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    write_csv_output(combined, path)


if __name__ == "__main__":
    raise SystemExit(main())
