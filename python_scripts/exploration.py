from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

CONFIG_DIR = Path(__file__).resolve().parent / "config"
FEATURE_SELECTION = json.loads((CONFIG_DIR / "boiler_feature_selection.json").read_text(encoding="utf-8"))
FAMILY_REDUCTION_ROWS = json.loads((CONFIG_DIR / "boiler_family_reduction.json").read_text(encoding="utf-8"))

TARGET_COLUMN = FEATURE_SELECTION["target_column"]
CANDIDATE_A_FEATURES = FEATURE_SELECTION["candidate_a_features"]
CANDIDATE_B_FEATURES = CANDIDATE_A_FEATURES + FEATURE_SELECTION["candidate_b_additions"]
CANDIDATE_C_FEATURES = CANDIDATE_B_FEATURES + FEATURE_SELECTION["candidate_c_additions"]
GRANULARITY_OPTIONS = FEATURE_SELECTION["granularity_options"]
CANDIDATE_B_PLOT_GROUPS = FEATURE_SELECTION["candidate_b_plot_groups"]
ASSUMED_PLOT_UNITS = FEATURE_SELECTION["assumed_plot_units"]

SMOOTHING_WINDOWS = {"1min": 60, "5min": 300}

HEATMAP_FAMILIES = {
    "pressure_state": [
        "PT_8313A.AV_0#",
        "PT_8313B.AV_0#",
        "PT_8313C.AV_0#",
        "PT_8313D.AV_0#",
        "PT_8313E.AV_0#",
        "PT_8313F.AV_0#",
        "PTCA_8322A.AV_0#",
        "PTCA_8324.AV_0#",
    ],
    "temperature_state": [
        "TE_8319A.AV_0#",
        "TE_8319B.AV_0#",
        "TE_8313B.AV_0#",
        "TE_8303.AV_0#",
        "TE_8304.AV_0#",
        "TE_8332A.AV_0#",
    ],
    "flow_state": [
        "FT_8301.AV_0#",
        "FT_8302.AV_0#",
        "FT_8306A.AV_0#",
        "FT_8306B.AV_0#",
        "YJJWSLL.AV_0#",
        "ZZQBCHLL.AV_0#",
    ],
    "oxygen_state": ["AIR_8301A.AV_0#", "AIR_8301B.AV_0#"],
    "fan_condition": ["YFJ3_AI.AV_0#", "YFJ3_ZD1.AV_0#", "YFJ3_ZD2.AV_0#"],
    "differential_pressure": ["SXLTCYZ.AV_0#", "SXLTCYY.AV_0#", "ZCLCCY.AV_0#", "YCLCCY.AV_0#"],
}


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------
# The output strategy is intentionally presentation-oriented:
# - Candidate B is the primary modeling dataset and is generated at every
#   granularity with matching plots.
# - Candidate C is kept as a raw control-aware reference only. Extra C
#   granularities are deferred until the modeling stage needs them.
# - CSV files are generated only when there is a plot or clear modeling artifact
#   tied to the same concept.

def write_boiler_outputs(data_folder: Path, output_folder: Path) -> None:
    data_dir = output_folder / "data"
    plots_dir = output_folder / "plots"
    reduced_plots_dir = plots_dir / "reduced_subsets"
    distribution_plots_dir = plots_dir / "distribution"
    outlier_plots_dir = plots_dir / "outliers"
    preprocessing_plots_dir = plots_dir / "preprocessing"

    for directory in [data_dir, reduced_plots_dir, distribution_plots_dir, outlier_plots_dir, preprocessing_plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    repaired = _load_repaired_boiler_frame(data_folder)
    repaired_path = data_dir / "boiler_repaired.csv"
    repaired.to_csv(repaired_path, index=False)

    subset_b_base = repaired[["date", TARGET_COLUMN, *CANDIDATE_B_FEATURES]].copy()
    subset_c_base = repaired[["date", TARGET_COLUMN, *CANDIDATE_C_FEATURES]].copy()
    subset_c_base.to_csv(data_dir / "subset_C.csv", index=False)
    _plot_control_loop(subset_c_base, reduced_plots_dir)
    _plot_heatmaps(repaired, plots_dir)

    subset_b_datasets = _write_subset_b_granularities(subset_b_base, data_dir, reduced_plots_dir)

    granularity_summary = _build_granularity_summary(subset_b_datasets)
    granularity_summary.to_csv(data_dir / "granularity_summary.csv", index=False)

    distribution_summary = _write_distribution_outputs(subset_b_datasets, data_dir, distribution_plots_dir)
    outlier_summary = _write_outlier_outputs(subset_b_datasets["subset_B_raw"], data_dir, outlier_plots_dir)
    smoothing_summary = _write_smoothing_differencing_outputs(subset_b_datasets, data_dir, preprocessing_plots_dir)

    _write_documentation_outputs(
        output_folder=output_folder,
        data_dir=data_dir,
        repaired_filename=repaired_path.name,
        granularity_summary=granularity_summary,
        distribution_summary=distribution_summary,
        outlier_summary=outlier_summary,
        smoothing_summary=smoothing_summary,
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_repaired_boiler_frame(data_folder: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_folder / "data.csv")
    autoreg = pd.read_csv(data_folder / "data_AutoReg.csv")
    return raw.combine_first(autoreg)


def _write_subset_b_granularities(
    subset_b_base: pd.DataFrame,
    data_dir: Path,
    plots_dir: Path,
) -> dict[str, pd.DataFrame]:
    datasets = {}

    for granularity_key, output_granularity in GRANULARITY_OPTIONS.items():
        id_key = f"subset_B_{granularity_key}"
        dataset = change_granularity(subset_b_base, output_granularity)
        datasets[id_key] = dataset
        dataset.to_csv(data_dir / f"{id_key}.csv", index=False)

        if granularity_key == "raw":
            dataset.to_csv(data_dir / "subset_B.csv", index=False)
            _plot_reduced_candidate_series(dataset, plots_dir, id_key)

    _plot_target_granularity_comparison(datasets, plots_dir)
    return datasets


def change_granularity(dataset: pd.DataFrame, output_granularity: str | None) -> pd.DataFrame:
    granular = dataset.copy()
    granular["date"] = pd.to_datetime(granular["date"], errors="coerce")
    granular = granular.dropna(subset=["date"]).sort_values("date")

    if output_granularity is None:
        return granular.reset_index(drop=True)

    return granular.set_index("date").resample(output_granularity).mean(numeric_only=True).reset_index()


def _build_granularity_summary(subset_b_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for id_key, dataset in subset_b_datasets.items():
        rows.append(
            {
                "id_key": id_key,
                "candidate": "subset_B",
                "granularity": id_key.replace("subset_B_", ""),
                "rows": len(dataset),
                "columns": len(dataset.columns),
                "start_date": dataset["date"].min(),
                "end_date": dataset["date"].max(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------
# Distribution summaries are per variable and per granularity. No global raw
# variance is used because sensor units and magnitudes differ.

def _write_distribution_outputs(
    subset_b_datasets: dict[str, pd.DataFrame],
    data_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    rows = []
    for id_key, dataset in subset_b_datasets.items():
        granularity = id_key.replace("subset_B_", "")
        for column in [col for col in dataset.columns if col != "date"]:
            rows.append(_distribution_summary_row(id_key, granularity, column, dataset[column]))

    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "distribution_summary.csv", index=False)
    _plot_target_distribution_by_granularity(subset_b_datasets, plots_dir)
    _plot_subset_b_standardized_boxplots(subset_b_datasets, plots_dir)
    return summary


def _distribution_summary_row(id_key: str, granularity: str, variable: str, values: pd.Series) -> dict[str, object]:
    clean_values = values.dropna()
    q1 = clean_values.quantile(0.25)
    q3 = clean_values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((clean_values < lower_bound) | (clean_values > upper_bound)).sum())

    return {
        "id_key": id_key,
        "granularity": granularity,
        "variable": variable,
        "count": int(clean_values.count()),
        "mean": clean_values.mean(),
        "std": clean_values.std(),
        "min": clean_values.min(),
        "q1": q1,
        "median": clean_values.median(),
        "q3": q3,
        "max": clean_values.max(),
        "iqr": iqr,
        "skewness": clean_values.skew(),
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_percent": 100 * outlier_count / len(clean_values) if len(clean_values) else 0,
    }


# ---------------------------------------------------------------------------
# Compact outlier analysis
# ---------------------------------------------------------------------------
# Detailed timestamp-level event tables are intentionally not generated here.
# They are too large for human review. Instead, the pipeline keeps compact
# visual summaries for the main raw Candidate B dataset.

def _write_outlier_outputs(
    subset_b_raw: pd.DataFrame,
    data_dir: Path,
    plots_dir: Path,
) -> dict[str, pd.DataFrame]:
    events = _locate_iqr_outliers(subset_b_raw)

    variable_counts = _build_outlier_variable_counts(events)
    variable_counts.to_csv(data_dir / "outlier_variable_counts.csv", index=False)

    top_windows = _build_top_outlier_windows(events)
    top_windows.to_csv(data_dir / "outlier_top_windows.csv", index=False)

    simultaneous = _build_top_simultaneous_outliers(events)
    simultaneous.to_csv(data_dir / "outlier_top_simultaneous_events.csv", index=False)

    _plot_outlier_variable_counts(variable_counts, plots_dir)
    _plot_outlier_top_windows(top_windows, plots_dir)
    _plot_simultaneous_outlier_counts(simultaneous, plots_dir)
    _plot_outlier_timeline(events, plots_dir)

    return {
        "variable_counts": variable_counts,
        "top_windows": top_windows,
        "simultaneous": simultaneous,
    }


def _locate_iqr_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    event_frames = []
    for column in [col for col in dataset.columns if col != "date"]:
        values = dataset[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (dataset[column] < lower_bound) | (dataset[column] > upper_bound)

        if not mask.any():
            continue

        events = dataset.loc[mask, ["date", column]].copy()
        events = events.rename(columns={column: "value"})
        events["variable"] = column
        events["direction"] = events["value"].apply(lambda value: "low" if value < lower_bound else "high")
        event_frames.append(events)

    if not event_frames:
        return pd.DataFrame(columns=["date", "variable", "value", "direction"])

    events = pd.concat(event_frames, ignore_index=True)
    events["date"] = pd.to_datetime(events["date"])
    return events.sort_values(["date", "variable"]).reset_index(drop=True)


def _build_outlier_variable_counts(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["variable", "direction"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .assign(total=lambda frame: frame.get("high", 0) + frame.get("low", 0))
        .sort_values("total", ascending=False)
    )


def _build_top_outlier_windows(events: pd.DataFrame) -> pd.DataFrame:
    windowed = events.copy()
    windowed["window_start"] = windowed["date"].dt.floor("5min")
    return (
        windowed.groupby("window_start")
        .agg(event_count=("variable", "size"), variable_count=("variable", "nunique"))
        .reset_index()
        .sort_values("event_count", ascending=False)
        .head(20)
    )


def _build_top_simultaneous_outliers(events: pd.DataFrame) -> pd.DataFrame:
    simultaneous = (
        events.groupby("date")
        .agg(
            variable_count=("variable", "nunique"),
            variables=("variable", lambda values: "; ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return simultaneous[simultaneous["variable_count"] >= 2].sort_values("variable_count", ascending=False).head(50)


# ---------------------------------------------------------------------------
# Smoothing and differencing diagnostics
# ---------------------------------------------------------------------------

def _write_smoothing_differencing_outputs(
    subset_b_datasets: dict[str, pd.DataFrame],
    data_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    rows = []
    for id_key, dataset in subset_b_datasets.items():
        granularity = id_key.replace("subset_B_", "")
        dataset = dataset.copy()
        dataset["date"] = pd.to_datetime(dataset["date"])
        target = dataset[TARGET_COLUMN]
        seconds_per_step = _infer_seconds_per_step(dataset)

        rows.append(_target_transform_summary_row(id_key, granularity, "original", "none", 1, target))

        smoothed_series = {}
        for window_label, window_seconds in SMOOTHING_WINDOWS.items():
            window_steps = max(1, round(window_seconds / seconds_per_step))
            if window_steps < 2:
                continue
            smoothed = target.rolling(window=window_steps, min_periods=window_steps).mean()
            smoothed_series[window_label] = smoothed
            rows.append(
                _target_transform_summary_row(
                    id_key,
                    granularity,
                    "trailing_rolling_mean",
                    window_label,
                    window_steps,
                    smoothed,
                )
            )

        first_difference = target.diff()
        rows.append(_target_transform_summary_row(id_key, granularity, "first_difference", "none", 1, first_difference))

        _plot_target_smoothing_comparison(dataset["date"], target, smoothed_series, plots_dir, id_key)
        _plot_target_first_difference(dataset["date"], first_difference, plots_dir, id_key)

    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "smoothing_differencing_summary.csv", index=False)
    _plot_smoothing_differencing_summary(summary, plots_dir)
    return summary


def _infer_seconds_per_step(dataset: pd.DataFrame) -> float:
    diffs = dataset["date"].diff().dropna().dt.total_seconds()
    return float(diffs.median()) if not diffs.empty else 5.0


def _target_transform_summary_row(
    id_key: str,
    granularity: str,
    transform: str,
    window_label: str,
    window_steps: int,
    values: pd.Series,
) -> dict[str, object]:
    clean_values = values.dropna()
    q1 = clean_values.quantile(0.25)
    q3 = clean_values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((clean_values < lower_bound) | (clean_values > upper_bound)).sum())

    return {
        "id_key": id_key,
        "granularity": granularity,
        "target": TARGET_COLUMN,
        "transform": transform,
        "window": window_label,
        "window_steps": window_steps,
        "count": int(clean_values.count()),
        "mean": clean_values.mean(),
        "std": clean_values.std(),
        "min": clean_values.min(),
        "median": clean_values.median(),
        "max": clean_values.max(),
        "iqr": iqr,
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_percent": 100 * outlier_count / len(clean_values) if len(clean_values) else 0,
    }


# ---------------------------------------------------------------------------
# Documentation outputs
# ---------------------------------------------------------------------------

def _write_documentation_outputs(
    output_folder: Path,
    data_dir: Path,
    repaired_filename: str,
    granularity_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    outlier_summary: dict[str, pd.DataFrame],
    smoothing_summary: pd.DataFrame,
) -> None:
    pd.DataFrame(FAMILY_REDUCTION_ROWS).to_csv(data_dir / "family_reduction_table.csv", index=False)
    (data_dir / "family_reduction_table.json").write_text(json.dumps(FAMILY_REDUCTION_ROWS, indent=2), encoding="utf-8")
    (output_folder / "feature_selection_note.json").write_text(json.dumps(FEATURE_SELECTION, indent=2), encoding="utf-8")

    output_files = {
        "exploration_findings.md": _build_exploration_markdown(repaired_filename),
        "granularity_analysis.md": _build_granularity_markdown(granularity_summary),
        "distribution_analysis.md": _build_distribution_markdown(distribution_summary),
        "outlier_analysis.md": _build_outlier_markdown(outlier_summary),
        "smoothing_differencing_analysis.md": _build_smoothing_differencing_markdown(smoothing_summary),
        "presentation_findings.md": _build_presentation_findings(
            granularity_summary,
            distribution_summary,
            outlier_summary,
            smoothing_summary,
        ),
    }

    for filename, content in output_files.items():
        output_path = output_folder / filename
        output_path.write_text(content, encoding="utf-8")
        print(f"Wrote {output_path}")


def _build_exploration_markdown(repaired_filename: str) -> str:
    return "\n".join(
        [
            "# Boiler Exploration Findings",
            "",
            "## Scope",
            "- The project is now focused on the Chinese boiler dataset only.",
            f"- The repaired working dataset is saved as `{repaired_filename}`.",
            f"- Forecasting target: `{TARGET_COLUMN}`.",
            "- Candidate B is the primary reduced dataset.",
            "- Candidate C is retained as a raw control-aware comparison for later modeling.",
        ]
    )


def _build_granularity_markdown(granularity_summary: pd.DataFrame) -> str:
    rows = [
        f"- `{row['id_key']}`: {row['rows']:,} rows from `{row['start_date']}` to `{row['end_date']}`."
        for row in granularity_summary.to_dict("records")
    ]
    return "\n".join(
        [
            "# Boiler Granularity Analysis",
            "",
            "## Decision",
            "- Candidate B is generated at `raw`, `30s`, `1min`, and `5min` granularities.",
            "- The raw cadence is 5 seconds and is already known to be regular.",
            "- Aggregation uses arithmetic mean because the selected variables are continuous process measurements.",
            "- Missing-window analysis is not repeated here; data quality already confirmed a constant 5-second cadence.",
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


def _build_distribution_markdown(distribution_summary: pd.DataFrame) -> str:
    target_rows = distribution_summary[distribution_summary["variable"] == TARGET_COLUMN][
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


def _build_outlier_markdown(outlier_summary: dict[str, pd.DataFrame]) -> str:
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


def _build_smoothing_differencing_markdown(smoothing_summary: pd.DataFrame) -> str:
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


def _build_presentation_findings(
    granularity_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    outlier_summary: dict[str, pd.DataFrame],
    smoothing_summary: pd.DataFrame,
) -> str:
    target_raw = distribution_summary[
        (distribution_summary["id_key"] == "subset_B_raw") & (distribution_summary["variable"] == TARGET_COLUMN)
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
            f"- The target is `{TARGET_COLUMN}`, interpreted as upper furnace chamber temperature.",
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_reduced_candidate_series(frame: pd.DataFrame, plots_dir: Path, id_key: str) -> None:
    for group_name, columns in CANDIDATE_B_PLOT_GROUPS.items():
        available_columns = [column for column in columns if column in frame.columns]
        if not available_columns:
            continue

        fig, axis = plt.subplots(figsize=(14, 5))
        for column in available_columns:
            axis.plot(frame["date"], frame[column], linewidth=1.0, label=column)
        axis.set_title(f"{id_key} - {group_name.replace('_', ' ').title()}")
        axis.set_xlabel("date")
        axis.set_ylabel(ASSUMED_PLOT_UNITS.get(group_name, "Sensor value"))
        axis.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{id_key}_{group_name}.png", dpi=150)
        plt.close(fig)


def _plot_control_loop(frame: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(pd.to_datetime(frame["date"]), frame["TV_8329ZC.AV_0#"], color="#3366aa", linewidth=1)
    axes[0].set_title("subset_C_raw - Valve Control Signal")
    axes[0].set_ylabel("Valve (%)")
    axes[1].plot(pd.to_datetime(frame["date"]), frame["YJJWSLL.AV_0#"], color="#aa6633", linewidth=1)
    axes[1].set_title("subset_C_raw - Desuperheating Water Flow")
    axes[1].set_ylabel("Flow")
    axes[1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(plots_dir / "subset_C_raw_control_loop.png", dpi=150)
    plt.close(fig)


def _plot_target_granularity_comparison(subset_b_datasets: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.ravel()

    for axis, (id_key, dataset) in zip(axes, subset_b_datasets.items()):
        axis.plot(dataset["date"], dataset[TARGET_COLUMN], linewidth=0.9)
        axis.set_title(id_key.replace("subset_B_", ""))
        axis.set_xlabel("date")
        axis.set_ylabel("Temp (assumed C)")

    fig.suptitle("Candidate B Target By Granularity")
    fig.tight_layout()
    fig.savefig(plots_dir / "subset_B_target_granularity_comparison.png", dpi=150)
    plt.close(fig)


def _plot_heatmaps(repaired: pd.DataFrame, plots_dir: Path) -> None:
    sampled = repaired.iloc[::20].drop(columns="date", errors="ignore")
    _plot_heatmap(sampled, plots_dir / "correlation_heatmap.png", "Boiler Correlation Heatmap")

    for family_name, columns in HEATMAP_FAMILIES.items():
        _plot_heatmap(
            sampled[columns],
            plots_dir / f"boiler_{family_name}_heatmap.png",
            f"Boiler {family_name.replace('_', ' ').title()} Heatmap",
        )


def _plot_heatmap(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(frame.corr(numeric_only=True), cmap="coolwarm", center=0, ax=axis)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_target_distribution_by_granularity(subset_b_datasets: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for axis, granularity_key in zip(axes, GRANULARITY_OPTIONS):
        id_key = f"subset_B_{granularity_key}"
        axis.hist(subset_b_datasets[id_key][TARGET_COLUMN].dropna(), bins=50, color="#2f6f8f", alpha=0.85)
        axis.set_title(granularity_key)
        axis.set_xlabel("Temp (assumed C)")
        axis.set_ylabel("Frequency")

    fig.suptitle("subset_B Target Distribution By Granularity")
    fig.tight_layout()
    fig.savefig(plots_dir / "subset_B_target_distribution_by_granularity.png", dpi=150)
    plt.close(fig)


def _plot_subset_b_standardized_boxplots(subset_b_datasets: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    for id_key, dataset in subset_b_datasets.items():
        frame = dataset.drop(columns="date")
        standardized = (frame - frame.mean()) / frame.std()
        fig, axis = plt.subplots(figsize=(14, 6))
        standardized.boxplot(ax=axis, showfliers=False, rot=75)
        axis.set_title(f"{id_key} - Standardized Variable Distribution")
        axis.set_ylabel("Z-score")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{id_key}_standardized_boxplot.png", dpi=150)
        plt.close(fig)


def _plot_outlier_variable_counts(variable_counts: pd.DataFrame, plots_dir: Path) -> None:
    if variable_counts.empty:
        return

    fig, axis = plt.subplots(figsize=(10, 6))
    variable_counts.sort_values("total").plot(x="variable", y="total", kind="barh", ax=axis, color="#ba4e00")
    axis.set_title("subset_B_raw - IQR Outlier Count By Variable")
    axis.set_xlabel("Outlier event count")
    axis.set_ylabel("Variable")
    legend = axis.get_legend()
    if legend is not None:
        legend.remove()
    fig.tight_layout()
    fig.savefig(plots_dir / "outlier_variable_counts.png", dpi=150)
    plt.close(fig)


def _plot_outlier_top_windows(top_windows: pd.DataFrame, plots_dir: Path) -> None:
    if top_windows.empty:
        return

    frame = top_windows.copy()
    frame["label"] = frame["window_start"].astype(str)
    fig, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(data=frame, y="label", x="event_count", ax=axis, color="#b85c38")
    axis.set_title("Top 20 Five-Minute Outlier Windows")
    axis.set_xlabel("Outlier event count")
    axis.set_ylabel("Window start")
    fig.tight_layout()
    fig.savefig(plots_dir / "outlier_top_windows.png", dpi=150)
    plt.close(fig)


def _plot_simultaneous_outlier_counts(simultaneous: pd.DataFrame, plots_dir: Path) -> None:
    if simultaneous.empty:
        return

    counts = simultaneous["variable_count"].value_counts().sort_index().reset_index()
    counts.columns = ["simultaneous_variable_count", "timestamp_count"]
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="simultaneous_variable_count", y="timestamp_count", ax=axis, color="#11875d")
    axis.set_title("subset_B_raw - Strongest Simultaneous Outlier Events")
    axis.set_xlabel("Variables outlying at the same timestamp")
    axis.set_ylabel("Timestamp count")
    fig.tight_layout()
    fig.savefig(plots_dir / "outlier_simultaneous_counts.png", dpi=150)
    plt.close(fig)


def _plot_outlier_timeline(events: pd.DataFrame, plots_dir: Path) -> None:
    if events.empty:
        return

    timeline = events.copy()
    timeline["window_start"] = timeline["date"].dt.floor("5min")
    timeline = timeline.groupby("window_start").size().reset_index(name="event_count")
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(timeline["window_start"], timeline["event_count"], linewidth=1.2, color="#8a3ffc")
    axis.set_title("subset_B_raw - IQR Outlier Events Per 5-Minute Window")
    axis.set_xlabel("date")
    axis.set_ylabel("Outlier event count")
    fig.tight_layout()
    fig.savefig(plots_dir / "outlier_timeline_5min.png", dpi=150)
    plt.close(fig)


def _plot_target_smoothing_comparison(
    dates: pd.Series,
    target: pd.Series,
    smoothed_series: dict[str, pd.Series],
    plots_dir: Path,
    id_key: str,
) -> None:
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(dates, target, linewidth=0.9, alpha=0.55, label="original")
    for window_label, smoothed in smoothed_series.items():
        axis.plot(dates, smoothed, linewidth=1.2, label=f"rolling mean {window_label}")
    axis.set_title(f"{id_key} - Target Smoothing Comparison")
    axis.set_xlabel("date")
    axis.set_ylabel("Temp (assumed C)")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{id_key}_target_smoothing_comparison.png", dpi=150)
    plt.close(fig)


def _plot_target_first_difference(
    dates: pd.Series,
    first_difference: pd.Series,
    plots_dir: Path,
    id_key: str,
) -> None:
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(dates, first_difference, linewidth=0.9, color="#4c72b0")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(f"{id_key} - Target First Difference")
    axis.set_xlabel("date")
    axis.set_ylabel("Delta temp")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{id_key}_target_first_difference.png", dpi=150)
    plt.close(fig)


def _plot_smoothing_differencing_summary(summary: pd.DataFrame, plots_dir: Path) -> None:
    frame = summary.copy()
    frame["transform_label"] = frame["transform"] + " (" + frame["window"] + ")"
    fig, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=frame, x="granularity", y="std", hue="transform_label", ax=axis)
    axis.set_title("Target Standard Deviation After Smoothing And Differencing")
    axis.set_xlabel("Granularity")
    axis.set_ylabel("Std")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "smoothing_differencing_std_summary.png", dpi=150)
    plt.close(fig)
