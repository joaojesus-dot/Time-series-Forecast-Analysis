from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extraction import DATASET_ANALYZERS
    from exploration import build_granularity_summary, summarize_target_transforms, summarize_variables
    from forecasting import run_forecasting_pipeline
    from plots import write_plot_specs
    from preprocessing import (
        build_feature_selection,
        build_feature_subset,
        build_granularity_versions,
        build_target_transform_series,
        load_repaired_boiler_frame,
    )
    from reports import write_documentation_outputs
else:
    from .extraction import DATASET_ANALYZERS
    from .exploration import build_granularity_summary, summarize_target_transforms, summarize_variables
    from .forecasting import run_forecasting_pipeline
    from .plots import write_plot_specs
    from .preprocessing import (
        build_feature_selection,
        build_feature_subset,
        build_granularity_versions,
        build_target_transform_series,
        load_repaired_boiler_frame,
    )
    from .reports import write_documentation_outputs


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
        forecasting_output,
        forecasting_reports_output,
        forecasting_plots_output,
        derived_data_output,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    family_reduction_rows = json.loads((CONFIG_DIR / "boiler_family_reduction.json").read_text(encoding="utf-8"))
    preprocessing_config = json.loads((CONFIG_DIR / "boiler_preprocessing.json").read_text(encoding="utf-8"))
    forecasting_config = json.loads((CONFIG_DIR / "boiler_forecasting.json").read_text(encoding="utf-8"))
    plot_config = json.loads((CONFIG_DIR / "boiler_plot_specs.json").read_text(encoding="utf-8"))
    feature_selection = build_feature_selection(family_reduction_rows)
    target_column = str(feature_selection["target_column"])
    candidate_b_features = list(feature_selection["candidate_b_features"])
    candidate_c_features = list(feature_selection["candidate_c_features"])
    granularity_options = preprocessing_config["granularity_options"]
    resampling_policy = preprocessing_config["resampling_policy"]

    # The orchestrator owns the KDD execution order. Lower-level modules only
    # transform data, calculate summaries, write plots, or write reports.
    quality_report = DATASET_ANALYZERS[dataset_name](args.data_dir / dataset_name)

    # Build the repaired working dataset once. All later summaries and plots use
    # this same source so every artifact is traceable to one cleaned frame.
    repaired = load_repaired_boiler_frame(args.data_dir / dataset_name)
    repaired_path = derived_data_output / "boiler_repaired.csv"
    write_csv_output(repaired, repaired_path)

    # Candidate B is the main forecasting feature set. Candidate C adds control
    # signals and is kept as a control-aware comparison for later modeling.
    subset_b_base = build_feature_subset(repaired, target_column, candidate_b_features)
    subset_c_base = build_feature_subset(repaired, target_column, candidate_c_features)
    write_csv_output(subset_c_base, derived_data_output / "subset_C.csv")

    # Generate all Candidate B granularities from the same base frame. This keeps
    # raw, 30s, 1min, and 5min datasets aligned in schema and naming.
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

    # Smoothing and differencing are diagnostic transforms for ARIMA/MLP
    # preparation. They do not replace the level target by default.
    transform_series = build_target_transform_series(subset_b_datasets, target_column, plot_config["smoothing_windows"])
    smoothing_summary = summarize_target_transforms(transform_series, target_column)

    write_excel_workbook(
        tables_output / "pre_forecasting_summary.xlsx",
        {
            "granularity": granularity_summary,
            "target_distribution": distribution_summary[distribution_summary["variable"] == target_column],
            "target_transforms": smoothing_summary,
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

    write_documentation_outputs(
        reports_dir=reports_output,
        repaired_filename=repaired_path.as_posix(),
        target_column=target_column,
        feature_selection=feature_selection,
        quality_report=quality_report,
        resampling_policy=resampling_policy,
        granularity_summary=granularity_summary,
        distribution_summary=distribution_summary,
        smoothing_summary=smoothing_summary,
    )
    if not args.skip_forecasting:
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
        write_forecasting_summary_workbook(tables_output / "forecasting_summary.xlsx", forecasting_results)
    return 0


def prepare_output_tree(dataset_output: Path, dataset_data_dir: Path) -> None:
    """Remove obsolete generated folders from the previous output layout."""
    obsolete_paths = [
        dataset_output / "data",
        dataset_output / "forecasting",
        dataset_output / "plots",
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


def write_excel_workbook(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.xlsx")
    with pd.ExcelWriter(temp_path) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    temp_path.replace(output_path)


def write_forecasting_summary_workbook(
    output_path: Path,
    forecasting_results: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Collect small forecasting summary frames into one reviewable workbook."""
    sheets = {}
    for analysis_name, result_frames in forecasting_results.items():
        for frame_name, frame in result_frames.items():
            if frame_name == "forecasts" or frame.empty:
                continue
            sheets[f"{analysis_name}_{frame_name}"] = frame
    if sheets:
        write_excel_workbook(output_path, sheets)


def build_boiler_plot_specs(
    plot_config: dict[str, Any],
    target_column: str,
    subset_b_datasets: dict[str, Any],
    transform_series: list[dict[str, Any]],
    output_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    # Plot specs live in JSON because they describe *what* should be plotted.
    # This function only resolves runtime values that JSON cannot hold:
    # filesystem paths, selected dataset keys, target column, and transform series.
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


if __name__ == "__main__":
    raise SystemExit(main())
