from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
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
    from forecasting import run_forecasting_pipeline
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
    from .forecasting import run_forecasting_pipeline
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Write boiler-only KDD and forecasting outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-forecasting", action="store_true")
    args = parser.parse_args()

    dataset_name = "chinese_boiler_dataset"
    dataset_output = args.output_dir / dataset_name
    reports_output = dataset_output / "reports"
    tables_output = dataset_output / "tables"
    plots_output = dataset_output / "plots"
    exploratory_plots_output = plots_output / "exploratory"
    reduced_plots_output = plots_output / "reduced_subsets"
    pre_forecasting_plots_output = plots_output / "pre_forecasting"
    profiling_plots_output = pre_forecasting_plots_output / "target_profiling"
    correlation_plots_output = pre_forecasting_plots_output / "target_correlations"
    forecasting_output = dataset_output / "forecasting"
    forecasting_reports_output = reports_output / "forecasting"
    forecasting_plots_output = plots_output / "forecasting"
    derived_data_output = args.data_dir / dataset_name / "derived"

    prepare_output_tree(dataset_output, args.data_dir / dataset_name)
    for directory in [
        reports_output,
        tables_output,
        exploratory_plots_output,
        reduced_plots_output,
        pre_forecasting_plots_output,
        profiling_plots_output,
        correlation_plots_output,
        forecasting_output,
        forecasting_reports_output,
        forecasting_plots_output,
        derived_data_output,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    family_reduction_rows = json.loads((CONFIG_DIR / "boiler_family_reduction.json").read_text(encoding="utf-8"))
    preprocessing_config = json.loads((CONFIG_DIR / "boiler_preprocessing.json").read_text(encoding="utf-8"))
    forecasting_config = json.loads((CONFIG_DIR / "boiler_forecasting.json").read_text(encoding="utf-8"))
    history_root = dataset_output / str(forecasting_config.get("run_tracking", {}).get("history_root", "history"))
    plot_config = json.loads((CONFIG_DIR / "boiler_plot_specs.json").read_text(encoding="utf-8"))
    feature_selection = build_feature_selection(family_reduction_rows)
    target_column = str(feature_selection["target_column"])
    candidate_b_features = list(feature_selection["candidate_b_features"])
    candidate_c_features = list(feature_selection["candidate_c_features"])
    granularity_options = preprocessing_config["granularity_options"]
    resampling_policy = preprocessing_config["resampling_policy"]
    profiling_granularities = list(forecasting_config["experimental_protocol"].get("profiling_granularities", []))
    scaling_method = str(forecasting_config["experimental_protocol"]["scaling"]["method"])

    # Load the dataset quality summary and the repaired working frame.
    quality_report = DATASET_ANALYZERS[dataset_name](args.data_dir / dataset_name)
    quality_summary = build_boiler_quality_summary(args.data_dir / dataset_name)

    repaired = load_repaired_boiler_frame(args.data_dir / dataset_name)
    repaired_path = derived_data_output / "boiler_repaired.csv"
    write_csv_output(repaired, repaired_path)

    # Build the reduced feature subsets used by the reports and forecasts.
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

    granularity_summary = build_granularity_summary(subset_b_datasets, dataset_name="subset_B")
    distribution_summary = summarize_variables(subset_b_datasets)
    profiling_datasets = {
        id_key: dataset
        for id_key, dataset in subset_b_datasets.items()
        if id_key.rsplit("_", 1)[-1] in profiling_granularities
    }

    # Build the cached target transforms used by the preparation summaries and plots.
    transform_series = build_target_transform_series(subset_b_datasets, target_column, plot_config["smoothing_windows"])
    smoothing_summary = summarize_target_transforms(transform_series, target_column)
    autocorrelation_summary = build_target_autocorrelation_summary(
        profiling_datasets,
        target_column,
        granularity_options,
        str(resampling_policy["input_frequency"]),
    )
    stationarity_summary = build_stationarity_summary(profiling_datasets, target_column)
    correlation_summary = build_target_correlation_summary(profiling_datasets, target_column)

    write_table_bundle(
        tables_output / "pre_forecasting_summary",
        {
            "granularity": granularity_summary,
            "target_distribution": distribution_summary[distribution_summary["variable"] == target_column],
            "target_transforms": smoothing_summary,
            "target_autocorrelation": autocorrelation_summary,
            "target_stationarity": stationarity_summary,
            "target_correlations": correlation_summary,
        },
    )

    frames = {
        "repaired": repaired,
        "subset_C": subset_c_base,
        "smoothing_summary": smoothing_summary,
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

    write_documentation_outputs(
        reports_dir=reports_output,
        repaired_filename=repaired_path.as_posix(),
        target_column=target_column,
        feature_selection=feature_selection,
        quality_report=quality_report,
        quality_summary=quality_summary,
        resampling_policy=resampling_policy,
        scaling_method=scaling_method,
        granularity_summary=granularity_summary,
        distribution_summary=distribution_summary,
        smoothing_summary=smoothing_summary,
        autocorrelation_summary=autocorrelation_summary,
        stationarity_summary=stationarity_summary,
        correlation_summary=correlation_summary,
    )
    write_experiment_plan(
        output_path=forecasting_reports_output / "experiment_plan.md",
        target_column=target_column,
        forecasting_policy=forecasting_config,
        resampling_policy=resampling_policy,
    )
    if not args.skip_forecasting:
        run_metadata = build_run_metadata(forecasting_config)
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
            reports_dir=forecasting_reports_output,
            tables_dir=tables_output / "forecasting_summary",
            tables_workbook=tables_output / "forecasting_summary.xlsx",
            plots_dir=forecasting_plots_output,
            comparison_frame=forecasting_results["comparison"]["model_comparison"],
            catalog_frame=forecasting_results["comparison"]["model_catalog"],
        )
    return 0


def prepare_output_tree(dataset_output: Path, dataset_data_dir: Path) -> None:
    """Remove obsolete generated folders from the previous output layout."""
    obsolete_paths = [
        dataset_output / "data",
        dataset_output / "forecasting",
        dataset_output / "plots",
        dataset_output / "reports",
        dataset_output / "tables",
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
        dataset_data_dir / "derived",
        dataset_data_dir / "feature_selection_note.json",
    ]
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
    tables = {}
    for analysis_name, result_frames in forecasting_results.items():
        for frame_name, frame in result_frames.items():
            if frame_name in {"forecasts", "mlp_forecasts", "scaled_target"} or frame.empty:
                continue
            table_name = f"{analysis_name}_{frame_name}"
            if analysis_name == "univariate" and frame_name == "metrics":
                table_name = "univariate_arima_metrics"
            tables[table_name] = frame
    if tables:
        write_table_bundle(output_dir, tables)


def build_boiler_plot_specs(
    plot_config: dict[str, Any],
    target_column: str,
    subset_b_datasets: dict[str, Any],
    transform_series: list[dict[str, Any]],
    output_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    # Resolve the dataset keys, target column, transforms, and output paths used by each plot spec.
    dataset_keys = list(subset_b_datasets)
    return [
        _resolve_plot_spec(spec, plot_config, target_column, dataset_keys, transform_series, output_dirs)
        for spec in plot_config["plot_specs"]
    ]


def _resolve_plot_spec(
    spec: dict[str, Any],
    plot_config: dict[str, Any],
    target_column: str,
    dataset_keys: list[str],
    transform_series: list[dict[str, Any]],
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

    shutil.copytree(reports_dir, run_root / "reports" / "forecasting", dirs_exist_ok=True)
    shutil.copytree(tables_dir, run_root / "tables" / "forecasting_summary", dirs_exist_ok=True)
    if tables_workbook.exists():
        shutil.copy2(tables_workbook, run_root / "tables" / tables_workbook.name)
    if bool(run_tracking.get("copy_plots", True)) and plots_dir.exists():
        shutil.copytree(plots_dir, run_root / "plots" / "forecasting", dirs_exist_ok=True)

    manifest = {
        **run_metadata,
        "forecasting_policy": forecasting_policy,
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
        key_columns=["run_id", "model_key", "split", "granularity"],
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
