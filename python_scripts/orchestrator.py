from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extraction import DATASET_ANALYZERS
    from exploration import (
        build_granularity_summary,
        locate_iqr_outliers,
        summarize_outliers,
        summarize_target_transforms,
        summarize_variables,
    )
    from plots import write_plot_specs
    from preprocessing import (
        build_feature_subset,
        build_granularity_versions,
        build_target_transform_series,
        load_repaired_boiler_frame,
    )
    from reports import write_documentation_outputs
else:
    from .extraction import DATASET_ANALYZERS
    from .exploration import (
        build_granularity_summary,
        locate_iqr_outliers,
        summarize_outliers,
        summarize_target_transforms,
        summarize_variables,
    )
    from .plots import write_plot_specs
    from .preprocessing import (
        build_feature_subset,
        build_granularity_versions,
        build_target_transform_series,
        load_repaired_boiler_frame,
    )
    from .reports import write_documentation_outputs


CONFIG_DIR = Path(__file__).resolve().parent / "config"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write boiler-only KDD and physical-analysis outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    dataset_name = "chinese_boiler_dataset"
    dataset_output = args.output_dir / dataset_name
    data_output = dataset_output / "data"
    plots_output = dataset_output / "plots"
    reduced_plots_output = plots_output / "reduced_subsets"
    distribution_plots_output = plots_output / "distribution"
    outlier_plots_output = plots_output / "outliers"
    preprocessing_plots_output = plots_output / "preprocessing"

    # The orchestrator owns the KDD execution order. The lower-level modules only
    # transform data, calculate summaries, write plots, or write reports.
    for directory in [
        dataset_output,
        data_output,
        reduced_plots_output,
        distribution_plots_output,
        outlier_plots_output,
        preprocessing_plots_output,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    feature_selection = json.loads((CONFIG_DIR / "boiler_feature_selection.json").read_text(encoding="utf-8"))
    family_reduction_rows = json.loads((CONFIG_DIR / "boiler_family_reduction.json").read_text(encoding="utf-8"))
    plot_config = json.loads((CONFIG_DIR / "boiler_plot_specs.json").read_text(encoding="utf-8"))
    target_column = feature_selection["target_column"]
    candidate_a_features = feature_selection["candidate_a_features"]
    candidate_b_features = candidate_a_features + feature_selection["candidate_b_additions"]
    candidate_c_features = candidate_b_features + feature_selection["candidate_c_additions"]

    quality_report = DATASET_ANALYZERS[dataset_name](args.data_dir / dataset_name)
    quality_path = dataset_output / "data_quality.md"
    quality_path.write_text(quality_report, encoding="utf-8")
    print(f"Wrote {quality_path}")

    # Build the repaired working dataset once. All later summaries and plots use
    # this same source so that every artifact is traceable to one cleaned frame.
    repaired = load_repaired_boiler_frame(args.data_dir / dataset_name)
    repaired_path = data_output / "boiler_repaired.csv"
    repaired.to_csv(repaired_path, index=False)

    # Candidate B is the main forecasting feature set. Candidate C adds control
    # signals and is kept as a control-aware comparison for later modeling.
    subset_b_base = build_feature_subset(repaired, target_column, candidate_b_features)
    subset_c_base = build_feature_subset(repaired, target_column, candidate_c_features)
    subset_c_base.to_csv(data_output / "subset_C.csv", index=False)

    # Generate all Candidate B granularities from the same base frame. This keeps
    # raw, 30s, 1min, and 5min datasets aligned in schema and naming.
    subset_b_datasets = build_granularity_versions(
        subset_b_base,
        feature_selection["granularity_options"],
        dataset_name="subset_B",
    )
    for id_key, dataset in subset_b_datasets.items():
        dataset.to_csv(data_output / f"{id_key}.csv", index=False)
    subset_b_datasets["subset_B_raw"].to_csv(data_output / "subset_B.csv", index=False)

    granularity_summary = build_granularity_summary(subset_b_datasets, dataset_name="subset_B")
    granularity_summary.to_csv(data_output / "granularity_summary.csv", index=False)

    # Distribution and outlier analysis share the same IQR bounds. This avoids
    # recalculating thresholds and keeps the outlier reports consistent with the
    # distribution tables.
    distribution_summary = summarize_variables(subset_b_datasets)
    distribution_summary.to_csv(data_output / "distribution_summary.csv", index=False)

    outlier_events = locate_iqr_outliers(subset_b_datasets["subset_B_raw"], distribution_summary, "subset_B_raw")
    outlier_summary = summarize_outliers(outlier_events)
    outlier_summary["variable_counts"].to_csv(data_output / "outlier_variable_counts.csv", index=False)
    outlier_summary["top_windows"].to_csv(data_output / "outlier_top_windows.csv", index=False)
    outlier_summary["simultaneous"].to_csv(data_output / "outlier_top_simultaneous_events.csv", index=False)

    # Smoothing and differencing are diagnostic transforms for ARIMA/LSTM
    # preparation. They do not replace the level target unless modeling later
    # proves they improve forecasting.
    transform_series = build_target_transform_series(subset_b_datasets, target_column, plot_config["smoothing_windows"])
    smoothing_summary = summarize_target_transforms(transform_series, target_column)
    smoothing_summary.to_csv(data_output / "smoothing_differencing_summary.csv", index=False)

    frames = {
        "repaired": repaired,
        "subset_C": subset_c_base,
        "outlier_events": outlier_events,
        "outlier_variable_counts": outlier_summary["variable_counts"],
        "outlier_top_windows": outlier_summary["top_windows"],
        "outlier_simultaneous": outlier_summary["simultaneous"],
        "smoothing_summary": smoothing_summary,
        **subset_b_datasets,
    }

    output_dirs = {
        "plots": plots_output,
        "reduced_plots": reduced_plots_output,
        "distribution_plots": distribution_plots_output,
        "outlier_plots": outlier_plots_output,
        "preprocessing_plots": preprocessing_plots_output,
    }
    write_plot_specs(
        build_boiler_plot_specs(
            plot_config,
            feature_selection,
            target_column,
            subset_b_datasets,
            transform_series,
            output_dirs,
        ),
        frames,
    )

    write_documentation_outputs(
        output_folder=dataset_output,
        data_dir=data_output,
        repaired_filename=repaired_path.name,
        target_column=target_column,
        feature_selection=feature_selection,
        family_reduction_rows=family_reduction_rows,
        granularity_summary=granularity_summary,
        distribution_summary=distribution_summary,
        outlier_summary=outlier_summary,
        smoothing_summary=smoothing_summary,
    )
    return 0


def build_boiler_plot_specs(
    plot_config: dict[str, Any],
    feature_selection: dict[str, Any],
    target_column: str,
    subset_b_datasets: dict[str, Any],
    transform_series: list[dict[str, Any]],
    output_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    # Plot specs live in JSON because they describe *what* should be plotted.
    # This function only resolves runtime values that JSON cannot hold:
    # filesystem paths, selected dataset keys, target column, and transform series.
    dataset_keys = list(subset_b_datasets)
    specs = [
        _resolve_plot_spec(spec, feature_selection, target_column, dataset_keys, transform_series, output_dirs)
        for spec in plot_config["plot_specs"]
    ]

    for family_name, columns in plot_config["heatmap_families"].items():
        specs.append(
            _resolve_plot_spec(
                {
                    "kind": "heatmap",
                    "frame": "repaired",
                    "sample_step": 20,
                    "columns": columns,
                    "title": f"Boiler {family_name.replace('_', ' ').title()} Heatmap",
                    "output_path_ref": ["plots", f"boiler_{family_name}_heatmap.png"],
                },
                feature_selection,
                target_column,
                dataset_keys,
                transform_series,
                output_dirs,
            )
        )

    return specs


def _resolve_plot_spec(
    spec: dict[str, Any],
    feature_selection: dict[str, Any],
    target_column: str,
    dataset_keys: list[str],
    transform_series: list[dict[str, Any]],
    output_dirs: dict[str, Path],
) -> dict[str, Any]:
    resolved = dict(spec)

    if resolved.pop("groups_ref", None) == "candidate_b_plot_groups":
        resolved["groups"] = feature_selection["candidate_b_plot_groups"]
    if resolved.pop("units_ref", None) == "assumed_plot_units":
        resolved["units"] = feature_selection["assumed_plot_units"]
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
