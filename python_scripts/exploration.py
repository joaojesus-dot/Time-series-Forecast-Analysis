from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

BOILER_FAMILIES = {
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
    "differential_pressure": [
        "SXLTCYZ.AV_0#",
        "SXLTCYY.AV_0#",
        "ZCLCCY.AV_0#",
        "YCLCCY.AV_0#",
    ],
    "valve_control": ["TV_8329ZC.AV_0#"],
}

RELATIONSHIP_ROWS = [
    {
        "sensor_a": "FT_8301.AV_0#",
        "sensor_b": "AIR_8301A.AV_0#",
        "physical_law": "Primary/air-side flow influences flue-gas O2 through excess-air balance.",
        "expected_sign": "+",
        "expected_lag": "short",
        "confounders": "fuel/feed changes, excess-air control",
        "dimension_note": "Do not collapse with O2; weak raw correlation may be controller-masked.",
    },
    {
        "sensor_a": "FT_8302.AV_0#",
        "sensor_b": "AIR_8301B.AV_0#",
        "physical_law": "Secondary air affects combustion completion and downstream O2.",
        "expected_sign": "+",
        "expected_lag": "short",
        "confounders": "load regime, fuel variation, mixing asymmetry",
        "dimension_note": "Keep one representative air-flow variable and one O2 variable.",
    },
    {
        "sensor_a": "PT_8313C.AV_0#",
        "sensor_b": "PT_8313B.AV_0#",
        "physical_law": "Upper furnace pressure sensors share the same pressure field.",
        "expected_sign": "+",
        "expected_lag": "near-zero",
        "confounders": "local tap effects, sensor bias",
        "dimension_note": "This block is redundant; one furnace-pressure representative is enough.",
    },
    {
        "sensor_a": "PTCA_8322A.AV_0#",
        "sensor_b": "PTCA_8324.AV_0#",
        "physical_law": "Steam-side/container pressure pair measures the same pressure domain.",
        "expected_sign": "+",
        "expected_lag": "near-zero",
        "confounders": "sensor scaling, local dynamics",
        "dimension_note": "Keep one steam-side pressure representative.",
    },
    {
        "sensor_a": "TE_8319A.AV_0#",
        "sensor_b": "TE_8319B.AV_0#",
        "physical_law": "Left/right economizer outlet temperatures should move together under balanced heat exchange.",
        "expected_sign": "+",
        "expected_lag": "near-zero",
        "confounders": "left/right fouling, gas maldistribution",
        "dimension_note": "Use one as baseline, monitor asymmetry as a maintenance residual.",
    },
    {
        "sensor_a": "TE_8303.AV_0#",
        "sensor_b": "TE_8304.AV_0#",
        "physical_law": "Primary and secondary air-preheater outlet temperatures share gas-path thermal forcing.",
        "expected_sign": "+",
        "expected_lag": "short",
        "confounders": "different air circuits, damper positions",
        "dimension_note": "Related but not identical; keep at least one representative plus target-side temperature.",
    },
    {
        "sensor_a": "TV_8329ZC.AV_0#",
        "sensor_b": "YJJWSLL.AV_0#",
        "physical_law": "Desuperheater valve action and spray-water flow belong to the same control loop.",
        "expected_sign": "+",
        "expected_lag": "near-zero",
        "confounders": "controller tuning, saturation, load ramps",
        "dimension_note": "This pair is partly redundant; include carefully to avoid controller leakage.",
    },
    {
        "sensor_a": "YJJWSLL.AV_0#",
        "sensor_b": "TE_8332A.AV_0#",
        "physical_law": "More desuperheating water should reduce outlet steam temperature.",
        "expected_sign": "-",
        "expected_lag": "short-to-medium",
        "confounders": "steam load, controller compensation",
        "dimension_note": "Weak raw correlation does not invalidate the physics.",
    },
    {
        "sensor_a": "ZZQBCHLL.AV_0#",
        "sensor_b": "TE_8313B.AV_0#",
        "physical_law": "Higher steam-production load couples to furnace thermal state.",
        "expected_sign": "+",
        "expected_lag": "medium",
        "confounders": "air/fuel control, soot/fouling, load regime",
        "dimension_note": "Useful cross-family signal for forecasting target temperature.",
    },
    {
        "sensor_a": "YFJ3_AI.AV_0#",
        "sensor_b": "SXLTCYZ.AV_0#",
        "physical_law": "Higher gas-path resistance should increase ID-fan duty/current.",
        "expected_sign": "+",
        "expected_lag": "short",
        "confounders": "fan control loop, mechanical efficiency, operating mode",
        "dimension_note": "Do not rely on raw correlation alone; lag and regime matter.",
    },
    {
        "sensor_a": "YFJ3_AI.AV_0#",
        "sensor_b": "YFJ3_ZD2.AV_0#",
        "physical_law": "Motor load and bearing vibration are linked through fan operating state and condition.",
        "expected_sign": "+",
        "expected_lag": "short",
        "confounders": "mechanical fault mode, imbalance, sensor placement",
        "dimension_note": "Keep current plus one vibration channel for maintenance-aware modeling.",
    },
    {
        "sensor_a": "ZCLCCY.AV_0#",
        "sensor_b": "YCLCCY.AV_0#",
        "physical_law": "Left/right layer differential pressures should move together under balanced gas flow.",
        "expected_sign": "+",
        "expected_lag": "near-zero",
        "confounders": "blockage, gas leakage, duct imbalance",
        "dimension_note": "One representative is enough for forecasting; asymmetry is useful for maintenance monitoring.",
    },
]

TARGET_COLUMN = "TE_8313B.AV_0#"

CANDIDATE_A_FEATURES = [
    "PT_8313C.AV_0#",
    "PTCA_8322A.AV_0#",
    "TE_8319A.AV_0#",
    "TE_8303.AV_0#",
    "ZZQBCHLL.AV_0#",
    "AIR_8301A.AV_0#",
    "YFJ3_AI.AV_0#",
    "SXLTCYZ.AV_0#",
]

CANDIDATE_B_ADDITIONS = [
    "TE_8332A.AV_0#",
    "FT_8301.AV_0#",
    "FT_8306A.AV_0#",
    "YFJ3_ZD2.AV_0#",
    "ZCLCCY.AV_0#",
]

CANDIDATE_C_ADDITIONS = [
    "TV_8329ZC.AV_0#",
    "YJJWSLL.AV_0#",
]

CANDIDATE_B_FEATURES = CANDIDATE_A_FEATURES + CANDIDATE_B_ADDITIONS
CANDIDATE_C_FEATURES = CANDIDATE_B_FEATURES + CANDIDATE_C_ADDITIONS

FAMILY_REDUCTION_ROWS = [
    {
        "family": "Target",
        "original_variables": TARGET_COLUMN,
        "observed_structure": "Primary forecasting target selected from the furnace thermal state.",
        "representative_variables": TARGET_COLUMN,
        "reason_for_keeping": "Temperature in the upper furnace chamber is thermally meaningful and connected to flow, fan, pressure, and load behavior.",
        "reason_for_dropping_others": "Not applicable; this variable is the prediction target and is included in all modeling subsets.",
        "candidate_membership": "A, B, C as target",
    },
    {
        "family": "Upper furnace pressure",
        "original_variables": "PT_8313A.AV_0#; PT_8313B.AV_0#; PT_8313C.AV_0#; PT_8313D.AV_0#; PT_8313E.AV_0#; PT_8313F.AV_0#",
        "observed_structure": "Strong redundant pressure-domain block.",
        "representative_variables": "PT_8313C.AV_0#",
        "reason_for_keeping": "Preserves upper-furnace pressure information while avoiding six highly similar channels.",
        "reason_for_dropping_others": "Other PT_8313 sensors observe the same pressure field and add redundancy for forecasting.",
        "candidate_membership": "A, B, C",
    },
    {
        "family": "Steam/container pressure",
        "original_variables": "PTCA_8322A.AV_0#; PTCA_8324.AV_0#",
        "observed_structure": "Coherent paired pressure domain.",
        "representative_variables": "PTCA_8322A.AV_0#",
        "reason_for_keeping": "Keeps steam-side pressure context as a separate domain from furnace pressure.",
        "reason_for_dropping_others": "PTCA_8324 is the paired counterpart and is redundant for the reduced forecasting set.",
        "candidate_membership": "A, B, C",
    },
    {
        "family": "Economizer temperature",
        "original_variables": "TE_8319A.AV_0#; TE_8319B.AV_0#",
        "observed_structure": "Left/right economizer outlet temperature pair.",
        "representative_variables": "TE_8319A.AV_0#",
        "reason_for_keeping": "Provides upstream flue-gas/economizer thermal context.",
        "reason_for_dropping_others": "TE_8319B is retained conceptually for asymmetry checks but removed from the reduced forecasting input set.",
        "candidate_membership": "A, B, C",
    },
    {
        "family": "Air-preheater and outlet thermal state",
        "original_variables": "TE_8303.AV_0#; TE_8304.AV_0#; TE_8332A.AV_0#",
        "observed_structure": "Temperature family is not one fully redundant block; outlet steam temperature is more isolated.",
        "representative_variables": "TE_8303.AV_0#; TE_8332A.AV_0#",
        "reason_for_keeping": "TE_8303 preserves air-preheater thermal context; TE_8332A is added in Candidate B to retain downstream steam-temperature behavior.",
        "reason_for_dropping_others": "TE_8304 is omitted from the reduced sets to avoid over-representing similar air-preheater behavior.",
        "candidate_membership": "TE_8303 in A/B/C; TE_8332A in B/C",
    },
    {
        "family": "Flow and load",
        "original_variables": "FT_8301.AV_0#; FT_8302.AV_0#; FT_8306A.AV_0#; FT_8306B.AV_0#; ZZQBCHLL.AV_0#",
        "observed_structure": "Flow variables contain more than one mechanism: fan flow, return-air flow, and compensated steam flow.",
        "representative_variables": "ZZQBCHLL.AV_0#; FT_8301.AV_0#; FT_8306A.AV_0#",
        "reason_for_keeping": "ZZQBCHLL represents steam-production load; FT_8301 and FT_8306A are added in Candidate B to preserve distinct flow mechanisms.",
        "reason_for_dropping_others": "FT_8302 and FT_8306B are omitted to keep the set compact and avoid paired-flow redundancy.",
        "candidate_membership": "ZZQBCHLL in A/B/C; FT_8301 and FT_8306A in B/C",
    },
    {
        "family": "Oxygen",
        "original_variables": "AIR_8301A.AV_0#; AIR_8301B.AV_0#",
        "observed_structure": "Strong left/right oxygen pair.",
        "representative_variables": "AIR_8301A.AV_0#",
        "reason_for_keeping": "Keeps excess-air/combustion information with one representative channel.",
        "reason_for_dropping_others": "AIR_8301B is redundant for forecasting inputs but remains meaningful for asymmetry diagnosis.",
        "candidate_membership": "A, B, C",
    },
    {
        "family": "Fan condition",
        "original_variables": "YFJ3_AI.AV_0#; YFJ3_ZD1.AV_0#; YFJ3_ZD2.AV_0#",
        "observed_structure": "Fan current and vibration do not collapse into one fully redundant group.",
        "representative_variables": "YFJ3_AI.AV_0#; YFJ3_ZD2.AV_0#",
        "reason_for_keeping": "YFJ3_AI captures fan duty/current; YFJ3_ZD2 is added in Candidate B to preserve mechanical-condition context.",
        "reason_for_dropping_others": "YFJ3_ZD1 is omitted to avoid carrying both vibration channels in the reduced set.",
        "candidate_membership": "YFJ3_AI in A/B/C; YFJ3_ZD2 in B/C",
    },
    {
        "family": "Differential pressure",
        "original_variables": "SXLTCYZ.AV_0#; SXLTCYY.AV_0#; ZCLCCY.AV_0#; YCLCCY.AV_0#",
        "observed_structure": "Layer differential pressures include paired structure and possible gas-path resistance information.",
        "representative_variables": "SXLTCYZ.AV_0#; ZCLCCY.AV_0#",
        "reason_for_keeping": "SXLTCYZ gives upper/lower hearth differential-pressure context; ZCLCCY is added in Candidate B to preserve a second gas-path resistance view.",
        "reason_for_dropping_others": "SXLTCYY and YCLCCY are omitted as paired counterparts in the reduced forecasting set.",
        "candidate_membership": "SXLTCYZ in A/B/C; ZCLCCY in B/C",
    },
    {
        "family": "Desuperheating control loop",
        "original_variables": "TV_8329ZC.AV_0#; YJJWSLL.AV_0#",
        "observed_structure": "Control-loop pair rather than purely passive process measurements.",
        "representative_variables": "TV_8329ZC.AV_0#; YJJWSLL.AV_0#",
        "reason_for_keeping": "Candidate C tests whether control-signal information improves forecasting.",
        "reason_for_dropping_others": "Excluded from Candidates A and B to keep the base physical set independent from explicit controller action.",
        "candidate_membership": "C only",
    },
]


def write_boiler_outputs(data_folder: Path, output_folder: Path) -> None:
    plots_dir = output_folder / "plots"
    data_dir = output_folder / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    repaired = _load_repaired_boiler_frame(data_folder)
    repaired_path = data_dir / "boiler_repaired.csv"
    repaired.to_csv(repaired_path, index=False)

    subset_B = repaired[["date", TARGET_COLUMN, *CANDIDATE_B_FEATURES]].copy()
    subset_C = repaired[["date", TARGET_COLUMN, *CANDIDATE_C_FEATURES]].copy()
    subset_B.to_csv(data_dir / "subset_B.csv", index=False)
    subset_C.to_csv(data_dir / "subset_C.csv", index=False)

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

    sampled = repaired.iloc[::20].copy()
    sampled["date"] = pd.to_datetime(sampled["date"], errors="coerce")

    _plot_family_series(sampled, plots_dir)
    _plot_family_heatmaps(sampled, plots_dir)
    _plot_heatmap(sampled.drop(columns="date"), plots_dir / "correlation_heatmap.png", "Boiler Correlation Heatmap")

    exploration_path = output_folder / "exploration_findings.md"
    exploration_path.write_text(_build_exploration_markdown(repaired_path.name), encoding="utf-8")
    print(f"Wrote {exploration_path}")

    relationship_frame = pd.DataFrame(RELATIONSHIP_ROWS)
    relationship_frame.to_csv(data_dir / "sensor_relationship_table.csv", index=False)

    family_reduction_frame = pd.DataFrame(FAMILY_REDUCTION_ROWS)
    family_reduction_frame.to_csv(data_dir / "family_reduction_table.csv", index=False)

    physical_path = output_folder / "physical_analysis.md"
    physical_path.write_text(_build_physical_analysis_markdown(), encoding="utf-8")
    print(f"Wrote {physical_path}")

    feature_selection_path = output_folder / "feature_selection_note.md"
    feature_selection_path.write_text(_build_feature_selection_markdown(), encoding="utf-8")
    print(f"Wrote {feature_selection_path}")


def _load_repaired_boiler_frame(data_folder: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_folder / "data.csv")
    autoreg = pd.read_csv(data_folder / "data_AutoReg.csv")
    return raw.combine_first(autoreg)


def _build_exploration_markdown(repaired_filename: str) -> str:
    return "\n".join(
        [
            "# Boiler Exploration Findings",
            "",
            "## KDD Focus",
            "- The project is now boiler-only, moving from dataset screening to a domain-driven KDD process.",
            "- The repaired working dataset is built from `data.csv` plus the 30 values filled by `data_AutoReg.csv`.",
            "- Variable families are interpreted through the boiler diagram before feature reduction decisions are made.",
            "",
            "## Findings",
            f"- The repaired dataset was saved to `{repaired_filename}` for downstream modeling.",
            "- Timestamp spacing is constant across the full series: the only observed gap is 5 seconds.",
            "- The heatmaps support clear pressure domains, left/right paired sensors, and control-loop structure.",
            "- Raw correlation is not enough for maintenance-oriented interpretation because control action, lag, and operating regime can hide simple physical links.",
            "- Family-level heatmaps are the right tool for representative-variable selection before LSTM experiments on the original and aggregated datasets.",
            "",
            "## Dimensionality Reduction Guidance",
            "- Upper furnace pressures A-F form one redundant block; keep one representative for forecasting inputs.",
            "- Steam-side pressures `PTCA_8322A` and `PTCA_8324` form a second redundant pair; keep one representative there as well.",
            "- The oxygen pair is nearly identical, so one variable is enough for the forecasting feature set.",
            "- Left/right paired sensors should not be discarded blindly: one representative supports forecasting, while left/right mismatch can be retained separately for physical diagnostics.",
        ]
    )


def _build_feature_selection_markdown() -> str:
    candidate_a = ", ".join(f"`{column}`" for column in CANDIDATE_A_FEATURES)
    candidate_b_additions = ", ".join(f"`{column}`" for column in CANDIDATE_B_ADDITIONS)
    candidate_c_additions = ", ".join(f"`{column}`" for column in CANDIDATE_C_ADDITIONS)

    return "\n".join(
        [
            "# Boiler Feature Selection Note",
            "",
            "## Objective",
            f"The forecasting target is `{TARGET_COLUMN}`, the temperature in the upper part of the furnace chamber. The feature-selection process is designed to reduce dimensionality while preserving the main physical domains of the boiler: pressure, temperature, flow, oxygen, fan condition, differential pressure, and control action.",
            "",
            "The selection is not based only on raw correlation. Boiler operation is controlled, and control loops can hide simple thermodynamic relationships. Therefore, the feature sets are defined in stages: first from physical structure, then from heatmap evidence, and finally from explicit control-loop information.",
            "",
            "## Selection Principles",
            "- Redundant sensors are compressed when they observe the same physical domain.",
            "- At least one representative is preserved for each relevant boiler segment.",
            "- Left/right pairs are reduced for forecasting, but asymmetry remains important for maintenance interpretation.",
            "- Control-loop variables are separated from the base physical set because they may improve forecasting while changing the interpretation of the model.",
            "",
            "## Candidate A: Physical-Only Reduced Dataset",
            "Candidate A is the dataset that would be selected using only the physical analysis and the boiler diagram. It keeps one representative per major physical segment and avoids variables that mostly duplicate the same measurement domain.",
            "",
            f"Candidate A columns, excluding `date` and the target `{TARGET_COLUMN}`:",
            "",
            candidate_a,
            "",
            "Candidate A is technically defensible because it preserves the core boiler structure with minimal redundancy: one upper-furnace pressure representative, one steam-side pressure representative, one economizer temperature, one air-preheater temperature, one steam-load flow variable, one oxygen variable, one fan-current variable, and one differential-pressure variable.",
            "",
            "## Candidate B: Heatmap-Informed Physical Dataset",
            "The family heatmaps show that some variables do not follow the simplest expected physical relationships. This does not invalidate the physical interpretation. In a controlled boiler, actuator action, operating regimes, time lags, load changes, and safety constraints can weaken or mask direct static correlations.",
            "",
            "For this reason, Candidate B is selected as the main working dataset. It keeps all Candidate A variables and adds variables that preserve distinct heatmap-supported behavior not captured by the minimum physical set.",
            "",
            "Variables added in Candidate B:",
            "",
            candidate_b_additions,
            "",
            "The added variables have specific roles. `TE_8332A.AV_0#` preserves downstream outlet steam-temperature behavior. `FT_8301.AV_0#` and `FT_8306A.AV_0#` retain additional flow mechanisms beyond compensated steam flow. `YFJ3_ZD2.AV_0#` adds mechanical-condition information not fully represented by fan current. `ZCLCCY.AV_0#` adds a second differential-pressure view of gas-path resistance.",
            "",
            "Candidate B is therefore the preferred base dataset for forecasting experiments. It is still reduced and physically interpretable, but it is less aggressive than Candidate A and better aligned with the observed multivariate structure.",
            "",
            "## Candidate C: Control-Aware Forecasting Dataset",
            "Candidate C is a second-level forecasting dataset. It keeps all Candidate B variables and adds the control-loop variables associated with desuperheating and outlet steam-temperature regulation.",
            "",
            "Variables added in Candidate C:",
            "",
            candidate_c_additions,
            "",
            "The motivation for Candidate C is that the boiler operates in a controlled environment. Control signals actively keep the process within safety and operating limits, which can mask simple sensor-to-sensor relationships. Including these variables may improve forecast performance because the model receives information about controller action, not only passive process measurements.",
            "",
            "Candidate C must be interpreted carefully. If it outperforms Candidate B, the improvement may come from controller intelligence rather than from a better physical representation of the uncontrolled plant. For that reason, Candidate C is not the primary physical dataset; it is a control-aware forecasting comparison.",
            "",
            "## Final Selection Strategy",
            "- Candidate A documents the minimum feature set justified by first-principles physical segmentation.",
            "- Candidate B is the main reduced dataset for forecasting because it combines physical segmentation with heatmap evidence.",
            "- Candidate C is reserved for testing whether explicit control-loop information improves LSTM forecasting performance.",
            "",
            "## Generated Dataset Files",
            "- Candidate B is saved as `outputs/chinese_boiler_dataset/data/subset_B.csv`.",
            "- Candidate C is saved as `outputs/chinese_boiler_dataset/data/subset_C.csv`.",
            "- The family-level reduction rationale is saved as `outputs/chinese_boiler_dataset/data/family_reduction_table.csv`.",
        ]
    )


def _build_physical_analysis_markdown() -> str:
    return "\n".join(
        [
            "# Boiler Physical Analysis",
            "",
            "This document establishes the first-principles interpretation of the boiler before the KDD modeling stage. The correlation heatmaps are treated as supporting evidence rather than as the source of causality.",
            "",
            "## 1. Physical Sensor Network",
            "- `PT_8313A` to `PT_8313F` are treated as one upper-furnace pressure domain observed at different positions.",
            "- `PTCA_8322A` and `PTCA_8324` are treated as the steam/container pressure pair.",
            "- `TE_8319A/B`, `TE_8313B`, `TE_8303`, `TE_8304`, and `TE_8332A` are treated as the main thermal chain: economizer/flue gas, furnace, air-preheater outlet, and outlet steam temperature.",
            "- `FT_8301`, `FT_8302`, `FT_8306A/B`, `YJJWSLL`, and `ZZQBCHLL` are treated as the main flow path variables: air/return-air flow, spray-water flow, and compensated steam flow.",
            "- `AIR_8301A` and `AIR_8301B` are treated as the left/right oxygen pair.",
            "- `YFJ3_AI`, `YFJ3_ZD1`, and `YFJ3_ZD2` are treated as fan-condition variables: current plus vibration.",
            "- `SXLTCYZ`, `SXLTCYY`, `ZCLCCY`, and `YCLCCY` are treated as the differential-pressure field for gas-path resistance and imbalance.",
            "- `TV_8329ZC` together with `YJJWSLL` is treated as the desuperheating control loop.",
            "",
            "## 2. Expected Causal Links",
            "- Higher primary and secondary air flow is expected to increase O2 unless fuel/feed rises proportionally.",
            "- Excess air is expected to reduce thermal efficiency and lower furnace temperature through dilution.",
            "- Economizer and air-preheater outlet temperatures are expected to reflect both thermal driving force and flow rate.",
            "- Steam-production load is expected to couple to furnace temperature, gas-path behavior, and draft effort.",
            "- Desuperheater valve action and spray-water flow are expected to reduce outlet steam temperature.",
            "- Differential pressures are expected to rise with gas or air flow, while abnormal increases at similar flow suggest fouling or blockage.",
            "- Induced-draft fan current is expected to rise with draft duty, gas flow, and path resistance.",
            "- Left/right duplicate sensors are expected to move together; persistent asymmetry is physically meaningful.",
            "- Pressure and flow are expected to react faster than temperatures, so lag must be considered.",
            "",
            "## 3. Prior Thermodynamic Dependency Graph",
            "- `FT_8301/FT_8302/FT_8306A/FT_8306B` -> `AIR_8301A/B`, `PT_8313*`, `SXLTCYZ/SXLTCYY/ZCLCCY/YCLCCY`.",
            "- `TE_8313B` + gas-path state -> `TE_8319A/B`.",
            "- flue-gas thermal state -> `TE_8303/TE_8304`.",
            "- `ZZQBCHLL` + `TV_8329ZC/YJJWSLL` -> `TE_8332A`.",
            "- `SXLTCYZ/SXLTCYY/ZCLCCY/YCLCCY` -> `YFJ3_AI`.",
            "- mechanical condition -> `YFJ3_ZD1/YFJ3_ZD2`.",
            "",
            "## 4. Sensor-to-Sensor Relationship Table",
            "- The structured sensor-to-sensor relationship table is stored separately in `outputs/chinese_boiler_dataset/data/sensor_relationship_table.csv`.",
            "- That CSV is the operational reference because it is easier to filter, sort, and extend than a markdown table.",
            "",
            "## Physical Findings",
            "- The heatmaps support clear pressure domains: the upper furnace pressure block and the steam-side pressure pair are both physically coherent.",
            "- The heatmaps support clear left/right paired sensors in the economizer temperatures, oxygen pair, and layer differential pressures.",
            "- The desuperheating variables behave like a control loop, so their relationships should be interpreted through control action rather than simple raw correlation.",
            "- The gas-path, steam-load, and thermal variables are coupled, but not in a simple static way. Lag, operating regime, and control compensation matter.",
            "- The weaker correlations around outlet steam temperature, O2 versus air flow, and return-air behavior do not invalidate the thermodynamic picture; they indicate that the plant is under active control and is not behaving like an open-loop system.",
            "",
            "## Representative-Variable View",
            "- The upper furnace pressure sensors can be reduced to one representative, and the steam-side pressure pair can be reduced to one representative, without losing the main pressure-domain structure.",
            "- The full temperature family should not be collapsed to one variable. A representative economizer outlet sensor, an air-preheater outlet temperature, and the target-side temperature view should be preserved.",
            "- The oxygen pair can be reduced to one representative for forecasting inputs.",
            "- Fan current plus one vibration channel should be retained when maintenance-aware physical context is needed instead of pure compression.",
            "- Left/right asymmetry should be preserved as a diagnostic concept even when the forecasting feature set keeps only one representative per pair.",
        ]
    )


def _plot_family_series(frame: pd.DataFrame, plots_dir: Path) -> None:
    for family_name, columns in BOILER_FAMILIES.items():
        fig, axis = plt.subplots(figsize=(14, 5))
        for column in columns:
            axis.plot(frame["date"], frame[column], linewidth=1.0, label=column)
        axis.set_title(f"Boiler {family_name.replace('_', ' ').title()}")
        axis.set_xlabel("date")
        axis.legend(loc="upper right", fontsize=8, ncol=2 if len(columns) > 4 else 1)
        fig.tight_layout()
        fig.savefig(plots_dir / f"boiler_{family_name}.png", dpi=150)
        plt.close(fig)


def _plot_family_heatmaps(frame: pd.DataFrame, plots_dir: Path) -> None:
    for family_name, columns in BOILER_FAMILIES.items():
        if len(columns) < 2:
            continue
        _plot_heatmap(
            frame[columns],
            plots_dir / f"boiler_{family_name}_heatmap.png",
            f"Boiler {family_name.replace('_', ' ').title()} Heatmap",
        )


def _plot_heatmap(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    correlation = frame.corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, cmap="coolwarm", center=0, ax=axis)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
