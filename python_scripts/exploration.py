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


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
# Feature-selection rationale and family-reduction documentation live in JSON
# files so the script remains focused on data generation and plotting. The JSON
# files are also copied to the output folder as formal project artifacts.

FEATURE_SELECTION = json.loads((CONFIG_DIR / "boiler_feature_selection.json").read_text(encoding="utf-8"))
FAMILY_REDUCTION_ROWS = json.loads((CONFIG_DIR / "boiler_family_reduction.json").read_text(encoding="utf-8"))

TARGET_COLUMN = FEATURE_SELECTION["target_column"]
CANDIDATE_A_FEATURES = FEATURE_SELECTION["candidate_a_features"]
CANDIDATE_B_ADDITIONS = FEATURE_SELECTION["candidate_b_additions"]
CANDIDATE_C_ADDITIONS = FEATURE_SELECTION["candidate_c_additions"]
CANDIDATE_B_FEATURES = CANDIDATE_A_FEATURES + CANDIDATE_B_ADDITIONS
CANDIDATE_C_FEATURES = CANDIDATE_B_FEATURES + CANDIDATE_C_ADDITIONS

CANDIDATE_DATASETS = {
    "subset_B": CANDIDATE_B_FEATURES,
    "subset_C": CANDIDATE_C_FEATURES,
}

GRANULARITY_OPTIONS = FEATURE_SELECTION["granularity_options"]
CANDIDATE_B_PLOT_GROUPS = FEATURE_SELECTION["candidate_b_plot_groups"]
ASSUMED_PLOT_UNITS = FEATURE_SELECTION["assumed_plot_units"]


# ---------------------------------------------------------------------------
# Public pipeline entrypoint
# ---------------------------------------------------------------------------

def write_boiler_outputs(data_folder: Path, output_folder: Path) -> None:
    plots_dir = output_folder / "plots"
    data_dir = output_folder / "data"
    reduced_plots_dir = plots_dir / "reduced_subsets"

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    reduced_plots_dir.mkdir(parents=True, exist_ok=True)

    repaired = _load_repaired_boiler_frame(data_folder)
    repaired_path = data_dir / "boiler_repaired.csv"
    repaired.to_csv(repaired_path, index=False)

    granularity_summary = _write_candidate_granularity_outputs(
        repaired,
        data_dir,
        reduced_plots_dir,
    )
    granularity_summary.to_csv(data_dir / "granularity_summary.csv", index=False)

    _write_gap_distribution(repaired, data_dir)
    _write_documentation_outputs(output_folder, data_dir, repaired_path.name)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
# The source data has a confirmed 5-second cadence. Resampling uses mean
# aggregation because the selected variables are continuous process signals.

def _load_repaired_boiler_frame(data_folder: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_folder / "data.csv")
    autoreg = pd.read_csv(data_folder / "data_AutoReg.csv")
    return raw.combine_first(autoreg)


def _write_candidate_granularity_outputs(
    repaired: pd.DataFrame,
    data_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    summary_rows = []

    for candidate_name, feature_columns in CANDIDATE_DATASETS.items():
        base_dataset = repaired[["date", TARGET_COLUMN, *feature_columns]].copy()

        for granularity_key, output_granularity in GRANULARITY_OPTIONS.items():
            id_key = f"{candidate_name}_{granularity_key}"
            granular_dataset = change_granularity(base_dataset, output_granularity)
            granular_dataset.to_csv(data_dir / f"{id_key}.csv", index=False)

            if granularity_key == "raw":
                granular_dataset.to_csv(data_dir / f"{candidate_name}.csv", index=False)

            summary_rows.append(_granularity_summary_row(id_key, candidate_name, granularity_key, granular_dataset))

            if candidate_name == "subset_B":
                _plot_reduced_candidate_series(granular_dataset, plots_dir, id_key)

    return pd.DataFrame(summary_rows)


def change_granularity(dataset: pd.DataFrame, output_granularity: str | None) -> pd.DataFrame:
    granular = dataset.copy()
    granular["date"] = pd.to_datetime(granular["date"], errors="coerce")
    granular = granular.dropna(subset=["date"]).sort_values("date")

    if output_granularity is None:
        return granular.reset_index(drop=True)

    return (
        granular.set_index("date")
        .resample(output_granularity)
        .mean(numeric_only=True)
        .reset_index()
    )


def _granularity_summary_row(
    id_key: str,
    candidate_name: str,
    granularity_key: str,
    dataset: pd.DataFrame,
) -> dict[str, object]:
    return {
        "id_key": id_key,
        "candidate": candidate_name,
        "granularity": granularity_key,
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "start_date": dataset["date"].min(),
        "end_date": dataset["date"].max(),
    }


def _write_gap_distribution(repaired: pd.DataFrame, data_dir: Path) -> None:
    gap_counts = (
        pd.to_datetime(repaired["date"], errors="coerce")
        .diff()
        .dropna()
        .dt.total_seconds()
        .value_counts()
        .sort_index()
    )
    pd.DataFrame(
        {"gap_seconds": gap_counts.index.astype(float), "count": gap_counts.values.astype(int)}
    ).to_csv(data_dir / "gap_distribution.csv", index=False)


# ---------------------------------------------------------------------------
# Documentation outputs
# ---------------------------------------------------------------------------
# JSON files preserve structured decisions for traceability. Markdown is kept
# only for short human-readable KDD and granularity notes.

def _write_documentation_outputs(output_folder: Path, data_dir: Path, repaired_filename: str) -> None:
    pd.DataFrame(FAMILY_REDUCTION_ROWS).to_csv(data_dir / "family_reduction_table.csv", index=False)
    (data_dir / "family_reduction_table.json").write_text(
        json.dumps(FAMILY_REDUCTION_ROWS, indent=2),
        encoding="utf-8",
    )
    (output_folder / "feature_selection_note.json").write_text(
        json.dumps(FEATURE_SELECTION, indent=2),
        encoding="utf-8",
    )

    output_files = {
        "exploration_findings.md": _build_exploration_markdown(repaired_filename),
        "granularity_analysis.md": _build_granularity_markdown(),
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
            "## KDD Focus",
            "- The project is now boiler-only, moving from dataset screening to a domain-driven KDD process.",
            "- The repaired working dataset is built from `data.csv` plus the 30 values filled by `data_AutoReg.csv`.",
            "- Feature reduction is based on boiler physics, family heatmaps, and explicit separation of control-loop variables.",
            "",
            "## Findings",
            f"- The repaired dataset was saved to `{repaired_filename}` for downstream traceability.",
            "- Timestamp spacing is constant across the full series: the only observed gap is 5 seconds.",
            "- Candidate B is the main reduced physical dataset for forecasting analysis.",
            "- Candidate C is the control-aware comparison dataset.",
            "- Feature-selection and family-reduction decisions are stored as JSON artifacts for traceability.",
        ]
    )


def _build_granularity_markdown() -> str:
    return "\n".join(
        [
            "# Boiler Granularity Analysis",
            "",
            "## Milestone 1: Modeling Scope",
            f"- Forecasting target: `{TARGET_COLUMN}`.",
            "- Primary modeling dataset: `subset_B`, the heatmap-informed reduced physical dataset.",
            "- Control-aware comparison dataset: `subset_C`, which adds explicit desuperheating control-loop variables.",
            "- The repaired full dataset remains available for traceability, but the modeling path is based on the reduced subsets.",
            "",
            "## Milestone 2: Temporal Granularity",
            "- The known raw sampling interval is 5 seconds.",
            "- Four temporal versions are generated for each reduced subset: `raw`, `30s`, `1min`, and `5min`.",
            "- The dynamic dataset identifier follows the pattern `<subset>_<granularity>`, for example `subset_B_raw`, `subset_B_30s`, `subset_B_1min`, and `subset_B_5min`.",
            "- Aggregation uses the arithmetic mean, which is appropriate at this stage because the selected variables are continuous process measurements.",
            "- Missing-window analysis is intentionally not included here because the source cadence has already been confirmed as regular.",
            "- Raw variance across all variables is not used as a summary metric because the sensors have different physical units and magnitudes.",
            "",
            "## Plot Scale And Units",
            "- The source metadata provides variable descriptions but does not explicitly provide engineering units.",
            "- Units are therefore treated as assumptions based on tag meaning, magnitude, and common Chinese industrial boiler conventions.",
            "- Temperature variables are interpreted as Celsius rather than Kelvin.",
            "- Oxygen is interpreted as percent O2.",
            "- Fan current is interpreted as amperes.",
            "- Vibration is interpreted as a source-native vibration unit, likely mm/s or micrometers.",
            "- Pressure and differential-pressure variables are interpreted as source-native Chinese DCS pressure units, likely Pa, kPa, or MPa depending on tag and magnitude.",
            "- Flow variables are interpreted as source-native flow units. Primary and return-air flows are likely m3/h or Nm3/h, while compensated steam flow may be t/h.",
            "- Plot y-axis labels are intentionally short; detailed unit assumptions are stored in `feature_selection_note.json`.",
            "- Flow variables are plotted separately because their magnitudes are very different: primary fan outlet flow is around 46,000, return-air flow is around 700, and compensated steam flow is around 60.",
            "",
            "## Generated Data Files",
            "- Candidate B: `subset_B_raw.csv`, `subset_B_30s.csv`, `subset_B_1min.csv`, `subset_B_5min.csv`.",
            "- Candidate C: `subset_C_raw.csv`, `subset_C_30s.csv`, `subset_C_1min.csv`, `subset_C_5min.csv`.",
            "- Backward-compatible raw aliases are also written as `subset_B.csv` and `subset_C.csv`.",
            "- A compact metadata summary is written to `granularity_summary.csv`.",
            "",
            "## Generated Plots",
            "- Time-series plots are generated for Candidate B at each granularity.",
            "- The plots are grouped by reduced physical segment and separated when units or magnitudes would distort interpretation.",
            "- Candidate B is plotted first because it is the main working dataset for the next KDD stage.",
        ]
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Grouped plots are useful only when the y-axis scale is comparable. Flow
# variables are intentionally split into separate plots because they have
# different magnitudes and would otherwise look artificially stationary.

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
        axis.set_ylabel(ASSUMED_PLOT_UNITS.get(group_name, "Sensor value, assumed source-native unit"))
        axis.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{id_key}_{group_name}.png", dpi=150)
        plt.close(fig)
