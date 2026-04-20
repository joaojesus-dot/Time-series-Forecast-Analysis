from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
# Reports are generated from already-computed dataframes. This keeps the
# narrative layer separate from the analysis layer and prevents hidden
# recalculation inside markdown generation.

def write_documentation_outputs(
    output_folder: Path,
    data_dir: Path,
    repaired_filename: str,
    target_column: str,
    feature_selection: dict[str, object],
    family_reduction_rows: list[dict[str, object]],
    resampling_policy: dict[str, object],
    granularity_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    outlier_summary: dict[str, pd.DataFrame],
    smoothing_summary: pd.DataFrame,
) -> None:
    pd.DataFrame(family_reduction_rows).to_csv(data_dir / "family_reduction_table.csv", index=False)
    (data_dir / "family_reduction_table.json").write_text(
        json.dumps(family_reduction_rows, indent=2),
        encoding="utf-8",
    )
    (output_folder / "feature_selection_note.json").write_text(
        json.dumps(feature_selection, indent=2),
        encoding="utf-8",
    )

    output_files = {
        "exploration_findings.md": build_exploration_markdown(repaired_filename, target_column),
        "granularity_analysis.md": build_granularity_markdown(granularity_summary, resampling_policy),
        "distribution_analysis.md": build_distribution_markdown(distribution_summary, target_column),
        "outlier_analysis.md": build_outlier_markdown(outlier_summary),
        "smoothing_differencing_analysis.md": build_smoothing_differencing_markdown(smoothing_summary),
        "presentation_findings.md": build_presentation_findings(
            target_column,
            distribution_summary,
            outlier_summary,
            smoothing_summary,
        ),
    }

    for filename, content in output_files.items():
        output_path = output_folder / filename
        output_path.write_text(content, encoding="utf-8")
        print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------
# These functions intentionally stay short and presentation-oriented. Detailed
# machine-readable decisions remain in JSON/CSV artifacts, while markdown files
# summarize what should be shown or defended in a review/presentation.

def build_exploration_markdown(repaired_filename: str, target_column: str) -> str:
    return "\n".join(
        [
            "# Boiler Exploration Findings",
            "",
            "## Scope",
            "- The project is now focused on the Chinese boiler dataset only.",
            f"- The repaired working dataset is saved as `{repaired_filename}`.",
            f"- Forecasting target: `{target_column}`.",
            "- Candidate B is the primary reduced dataset.",
            "- Candidate C is retained as a raw control-aware comparison for later modeling.",
        ]
    )


def build_granularity_markdown(granularity_summary: pd.DataFrame, resampling_policy: dict[str, object]) -> str:
    rows = [
        f"- `{row['id_key']}`: {row['rows']:,} rows from `{row['start_date']}` to `{row['end_date']}`."
        for row in granularity_summary.to_dict("records")
    ]
    column_aggregations = resampling_policy.get("column_aggregations", {})
    column_aggregation_lines = [
        f"- `{column}`: `{aggregation}`."
        for column, aggregation in sorted(column_aggregations.items())
    ]
    return "\n".join(
        [
            "# Boiler Granularity Analysis",
            "",
            "## Decision",
            "- Candidate B is generated at `raw`, `30s`, `1min`, and `5min` granularities.",
            "- The raw cadence is 5 seconds and is already known to be regular.",
            "- Resampled timestamps are right-labeled causal window endpoints.",
            f"- Windows are `{resampling_policy['closed']}`-closed and labeled on the `{resampling_policy['label']}` edge.",
            f"- Window origin is `{resampling_policy['origin']}` and incomplete edge windows are dropped: `{resampling_policy['drop_partial_windows']}`.",
            f"- Default aggregation is `{resampling_policy['default_aggregation']}`.",
            "- Missing-window analysis is not repeated here; data quality already confirmed a constant 5-second cadence.",
            "",
            "## Configured Column Aggregation Overrides",
            "- Overrides are applied only when the column is present in a resampled dataset.",
            *(column_aggregation_lines or ["- None."]),
            "",
            "## Generated Datasets",
            *rows,
            "",
            "## Unit Assumptions",
            "- Engineering units are not explicitly provided by the source metadata.",
            "- Temperature is treated as Celsius, oxygen as `% O2`, fan current as amperes, and flow/pressure variables as source-native Chinese DCS units inferred from tag meaning and magnitude.",
            "- Flow variables are plotted separately because primary fan flow, return-air flow, and steam flow have very different magnitudes.",
            "",
            "## Generated Plots",
            "- Raw Candidate B variables are plotted by physical segment for interpretation.",
            "- The target is plotted across all granularities in one comparison figure.",
            "- Candidate C is represented by one raw control-loop plot; extra Candidate C granularities are deferred until modeling.",
        ]
    )


def build_distribution_markdown(distribution_summary: pd.DataFrame, target_column: str) -> str:
    target_rows = distribution_summary[distribution_summary["variable"] == target_column][
        ["granularity", "mean", "std", "iqr_outlier_percent"]
    ]
    target_lines = [
        f"- `{row['granularity']}`: mean `{row['mean']:.2f}`, std `{row['std']:.2f}`, "
        f"IQR outliers `{row['iqr_outlier_percent']:.2f}%`."
        for row in target_rows.to_dict("records")
    ]

    top_raw = (
        distribution_summary[distribution_summary["id_key"] == "subset_B_raw"]
        .sort_values("iqr_outlier_percent", ascending=False)
        .head(6)
    )
    top_lines = [
        f"- `{row['variable']}`: `{row['iqr_outlier_percent']:.2f}%` IQR outliers."
        for row in top_raw.to_dict("records")
    ]

    return "\n".join(
        [
            "# Boiler Distribution Analysis",
            "",
            "## Target Distribution",
            *target_lines,
            "",
            "## Main Outlier Contributors In `subset_B_raw`",
            *top_lines,
            "",
            "## Interpretation",
            "- The target distribution is stable across granularities; aggregation changes the spread only slightly.",
            "- Cross-variable distribution comparisons should use standardized plots because sensor units and magnitudes differ.",
            "- Outlier counts are screening indicators, not treatment decisions.",
        ]
    )


def build_outlier_markdown(outlier_summary: dict[str, pd.DataFrame]) -> str:
    variable_counts = outlier_summary["variable_counts"].head(6)
    top_windows = outlier_summary["top_windows"].head(5)
    simultaneous = outlier_summary["simultaneous"].head(5)

    variable_lines = [
        f"- `{row['variable']}`: total `{row['total']}`, high `{row.get('high', 0)}`, "
        f"low `{row.get('low', 0)}`."
        for row in variable_counts.to_dict("records")
    ]
    window_lines = [
        f"- `{row['window_start']}`: `{row['event_count']}` events across `{row['variable_count']}` variables."
        for row in top_windows.to_dict("records")
    ]
    simultaneous_lines = [
        f"- `{row['date']}`: `{row['variable_count']}` simultaneous variables: `{row['variables']}`."
        for row in simultaneous.to_dict("records")
    ]

    return "\n".join(
        [
            "# Boiler Outlier Analysis",
            "",
            "## Key Finding",
            "- Outliers are not treated as bad data by default. The strongest events are clustered and multivariate, which is consistent with real operating transitions or control action.",
            "",
            "## Main Variables",
            *variable_lines,
            "",
            "## Highest-Density Windows",
            *window_lines,
            "",
            "## Strongest Simultaneous Events",
            *simultaneous_lines,
            "",
            "## Modeling Decision",
            "- Do not delete outliers automatically.",
            "- Keep multivariate clusters as real operating information.",
            "- Review isolated single-variable spikes separately if they affect model training.",
        ]
    )


def build_smoothing_differencing_markdown(smoothing_summary: pd.DataFrame) -> str:
    std_lines = [
        f"- `{row['id_key']}` `{row['transform']}` `{row['window']}`: std `{row['std']:.3f}`."
        for row in smoothing_summary.to_dict("records")
        if row["transform"] in {"original", "trailing_rolling_mean", "first_difference"}
    ]
    return "\n".join(
        [
            "# Boiler Smoothing And Differencing Analysis",
            "",
            "## Decision",
            "- Smoothing is not selected as a primary modeling target transformation at this stage.",
            "- Rolling smoothing only slightly reduces target spread and largely duplicates what temporal aggregation already provides.",
            "- First differencing remains relevant for ARIMA testing because it converts the target level into a change series centered near zero.",
            "",
            "## Summary Statistics",
            *std_lines,
            "",
            "## Modeling Implication",
            "- ARIMA should test both `d=0` and `d=1` instead of assuming differencing is required.",
            "- LSTM targets should remain in level form unless later validation results show a clear reason to predict differences.",
        ]
    )


def build_presentation_findings(
    target_column: str,
    distribution_summary: pd.DataFrame,
    outlier_summary: dict[str, pd.DataFrame],
    smoothing_summary: pd.DataFrame,
) -> str:
    target_raw = distribution_summary[
        (distribution_summary["id_key"] == "subset_B_raw") & (distribution_summary["variable"] == target_column)
    ].iloc[0].to_dict()
    top_outlier = outlier_summary["variable_counts"].iloc[0].to_dict()
    raw_smoothing = smoothing_summary[
        (smoothing_summary["id_key"] == "subset_B_raw") & (smoothing_summary["transform"] == "original")
    ].iloc[0].to_dict()
    raw_5min_smooth = smoothing_summary[
        (smoothing_summary["id_key"] == "subset_B_raw")
        & (smoothing_summary["transform"] == "trailing_rolling_mean")
        & (smoothing_summary["window"] == "5min")
    ].iloc[0].to_dict()

    return "\n".join(
        [
            "# Presentation Findings",
            "",
            "## Dataset Selection",
            "- The Chinese boiler dataset was selected because it has a regular 5-second cadence, clear physical structure, and only 30 missing cells, all repaired by `data_AutoReg.csv`.",
            "",
            "## Forecasting Target",
            f"- The target is `{target_column}`, interpreted as upper furnace chamber temperature.",
            f"- In the raw Candidate B dataset, the target mean is `{target_raw['mean']:.2f}` and standard deviation is `{target_raw['std']:.2f}`.",
            "",
            "## Feature Set",
            "- Candidate B is the main reduced feature set because it balances physical reasoning with observed heatmap structure.",
            "- Candidate C is retained as a control-aware comparison because boiler control signals may improve forecasting but change interpretation.",
            "",
            "## Granularity",
            "- Candidate B is generated at raw, 30-second, 1-minute, and 5-minute granularities.",
            "- Aggregation should be compared by validation performance rather than chosen visually.",
            "",
            "## Distribution And Outliers",
            f"- The largest outlier contributor in `subset_B_raw` is `{top_outlier['variable']}` "
            f"with `{top_outlier['total']}` IQR events.",
            "- Outlier clusters are multivariate, so automatic deletion is not justified.",
            "",
            "## Smoothing And Differencing",
            f"- Raw target std is `{raw_smoothing['std']:.2f}`; raw target with 5-minute trailing smoothing has std `{raw_5min_smooth['std']:.2f}`.",
            "- Smoothing does not add enough value to become the default target transformation.",
            "- First differencing should be tested through ARIMA orders rather than selected manually at this stage.",
        ]
    )
