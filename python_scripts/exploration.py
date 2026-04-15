from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


sns.set_theme(style="whitegrid")

METRO_FAMILIES = {
    "pressure_state": ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"],
    "thermal_state": ["Oil_temperature"],
    "load_state": ["Motor_current"],
    "control_state": ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level"],
    "flow_state": ["Caudal_impulses"],
}

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


def explore_metro(data_folder: Path, output_folder: Path) -> str:
    csv_path = data_folder / "MetroPT3(AirCompressor).csv"
    plots_dir, data_dir = _prepare_output_dirs(output_folder)
    _remove_files(
        [
            plots_dir / "correlation_heatmap.png",
            plots_dir / "metro_COMP_vs_DV_eletric.png",
            plots_dir / "metro_COMP_vs_MPG.png",
            plots_dir / "metro_TP2_vs_DV_eletric.png",
            plots_dir / "metro_TP2_vs_H1.png",
            plots_dir / "metro_TP3_vs_Reservoirs.png",
        ]
    )

    gap_counts = _gap_distribution_from_csv(csv_path, "timestamp")
    gap_frame = pd.DataFrame(
        {"gap_seconds": list(gap_counts.keys()), "count": list(gap_counts.values())}
    ).sort_values("gap_seconds")
    gap_frame.to_csv(data_dir / "gap_distribution.csv", index=False)

    sampled = _sample_csv(
        csv_path,
        usecols=["timestamp", *sum(METRO_FAMILIES.values(), [])],
        sample_step=20,
    )
    sampled["timestamp"] = pd.to_datetime(sampled["timestamp"], errors="coerce")
    sampled = sampled.dropna(subset=["timestamp"]).reset_index(drop=True)

    _plot_gap_histogram(
        gap_frame[gap_frame["gap_seconds"].between(8, 30)],
        plots_dir / "time_gap_histogram_8s_to_30s.png",
        "Metro Time-Gap Histogram Around 10s",
    )
    _plot_gap_distribution_full(
        gap_frame,
        plots_dir / "time_gap_histogram_full.png",
        "Metro Full Time-Gap Distribution",
    )
    _plot_family_series(sampled, "timestamp", METRO_FAMILIES, plots_dir, "metro")
    _plot_state_heatmap(
        sampled,
        "timestamp",
        plots_dir / "metro_state_heatmap.png",
        "Metro Normalized State Heatmap",
        max_time_points=600,
    )

    large_gap_sizes = int((gap_frame["gap_seconds"] > 30).sum())
    large_gap_events = int(gap_frame.loc[gap_frame["gap_seconds"] > 30, "count"].sum())
    metro_exploration_markdown = "\n".join(
        [
            "# Metro Exploration Findings",
            "",
            "## KDD Focus",
            "- Selected because it is multivariate, physically meaningful, and rich enough for autocorrelation and state-based forecasting.",
            "- Preprocessing for exploration used the original timestamps, full gap counting, and a 1-in-20 sample for visualization.",
            "- Transformation grouped variables by physical role: pressure, thermal, load, control, and flow.",
            "",
            "## Findings",
            f"- The dataset has {len(gap_frame):,} distinct timestamp gaps, so the cadence is not limited to only 3 values.",
            "- The nominal cadence is still about 10 seconds, with 9s to 13s representing most of the sampling jitter.",
            f"- There are {large_gap_events:,} gap events above 30 seconds across {large_gap_sizes:,} distinct large-gap sizes, which supports splitting the dataset into continuous segments before autocorrelation work.",
            "- The normalized state heatmap is more informative than pairwise correlation visuals for this dataset because the behavior is dominated by operating modes over time.",
            "",
            "## Forecasting Ideas",
            "- Forecast `TP3` or `Reservoirs` to model pneumatic availability and pressure stability.",
            "- Forecast `Motor_current` if the objective is compressor load and operating regime anticipation.",
            "- Forecast `Caudal_impulses` if the focus is downstream air demand rather than compressor internals.",
            "- For the first modeling pass, use continuous segments separated by larger outages and test lag structure inside each segment.",
        ]
    )
    return metro_exploration_markdown


def explore_garment(data_folder: Path, output_folder: Path) -> str:
    csv_path = data_folder / "merged_filtered_monotonic.csv"
    plots_dir, _ = _prepare_output_dirs(output_folder)
    _remove_files(
        [
            plots_dir / "target_output_sequence.png",
            plots_dir / "quality_histogram_0_to_100.png",
        ]
    )

    frame = pd.read_csv(csv_path)
    long_frame = _garment_long_frame(frame)
    _plot_garment_slot_profile(long_frame, plots_dir / "slot_profile_target_vs_output.png")
    _plot_garment_sampled_sequences(long_frame, plots_dir / "sampled_sequences.png")
    _plot_garment_cluster_profiles(long_frame, plots_dir / "clustered_output_profiles.png")
    _plot_garment_forecast_signal(long_frame, plots_dir / "early_to_final_signal.png")
    quality_percent = 100 * long_frame["output"] / long_frame["target"]
    quality_frame = long_frame.assign(quality_percent=quality_percent)
    _plot_garment_quality_by_slot(
        quality_frame,
        plots_dir / "quality_by_slot_boxplot.png",
    )
    quality_summary_markdown = _garment_quality_summary_markdown(quality_frame)

    garment_exploration_markdown = "\n".join(
        [
            "# Garment Exploration Findings",
            "",
            "## KDD Focus",
            "- Selected as a slot-sequence production problem rather than a calendar-time problem.",
            "- Preprocessing reshaped the wide table into ordered `(slot, target, output)` observations.",
            "- Transformation added a quality ratio defined as `output / target`.",
            "- In the forecasting plots, `final output` means the last observed slot output, i.e. `Output_11`.",
            "",
            "## Findings",
            "- This dataset behaves like repeated within-shift production trajectories.",
            "- Targets define the planned production curve, while outputs describe how each line or shift actually progressed through the slots.",
            "- Treating the slots as an ordered sequence is reasonable for exploratory forecasting, even without an explicit timestamp column.",
            "- Quality values above 100% exist, which means some outputs exceeded the planned target and should not be clipped away in modeling.",
            "- Slot-profile and per-slot quality plots are more interpretable here than a single flattened sequence plot.",
            "- Early-slot outputs already carry strong signal for final output, which supports a forecast-from-partial-shift framing.",
            "- The trajectory clusters separate low, medium, and high production patterns, which is useful for scenario-based forecasting.",
            "",
            "## Forecasting Ideas",
            "- Forecast the next slot output from the early slots.",
            "- Forecast the final output level from the first few slots of a sequence.",
            "- Forecast the final quality ratio or final target shortfall to support intervention during the shift.",
            "",
            "## Quality Five-Number Summary By Slot",
            "",
            quality_summary_markdown,
        ]
    )
    return garment_exploration_markdown


def explore_boiler(data_folder: Path, output_folder: Path) -> str:
    original_path = data_folder / "data.csv"
    autoreg_path = data_folder / "data_AutoReg.csv"
    plots_dir, data_dir = _prepare_output_dirs(output_folder)

    original = pd.read_csv(original_path)
    autoreg = pd.read_csv(autoreg_path)
    repaired = original.combine_first(autoreg)
    repaired_path = data_dir / "boiler_repaired.csv"
    repaired.to_csv(repaired_path, index=False)

    gap_counts = (
        pd.to_datetime(repaired["date"], errors="coerce")
        .diff()
        .dropna()
        .dt.total_seconds()
        .value_counts()
        .sort_index()
    )
    gap_frame = pd.DataFrame(
        {"gap_seconds": gap_counts.index.astype(float), "count": gap_counts.values.astype(int)}
    )
    gap_frame.to_csv(data_dir / "gap_distribution.csv", index=False)

    sampled = repaired.iloc[::20].copy()
    sampled["date"] = pd.to_datetime(sampled["date"], errors="coerce")
    _plot_family_series(sampled, "date", BOILER_FAMILIES, plots_dir, "boiler")
    _plot_family_heatmaps(sampled, BOILER_FAMILIES, plots_dir, "boiler")
    _plot_heatmap(
        sampled.drop(columns="date"),
        plots_dir / "correlation_heatmap.png",
        "Boiler Correlation Heatmap",
    )

    boiler_exploration_markdown = "\n".join(
        [
            "# Boiler Exploration Findings",
            "",
            "## KDD Focus",
            "- Selected because the timestamp cadence is regular and the process diagram gives direct physical context.",
            "- Preprocessing repaired the original dataset with `data_AutoReg.csv` to remove the 30 missing values in `YJJWSLL.AV_0#`.",
            "- Transformation grouped variables into pressure, temperature, flow, oxygen, fan-condition, differential-pressure, and valve-control families.",
            "",
            "## Findings",
            f"- The repaired dataset was saved to `{repaired_path.name}` for downstream use.",
            "- Timestamp spacing is constant across the full series: the only observed gap is 5 seconds.",
            "- The variable families align well with the boiler diagram, so this dataset supports both statistical and physical interpretation.",
            "- Family-specific heatmaps were generated to support representative-variable selection inside each physical group.",
            "",
            "## Forecasting Ideas",
            "- Forecast `TE_8332A.AV_0#` for boiler outlet steam temperature control.",
            "- Forecast `ZZQBCHLL.AV_0#` for compensated main steam flow demand.",
            "- Forecast `YJJWSLL.AV_0#` to study desuperheating water behavior after repairing the gaps.",
            "- Build models at a family level first to identify whether pressure, temperature, or flow variables contribute the strongest predictive signal.",
        ]
    )
    return boiler_exploration_markdown


def explore_beverage(data_folder: Path, output_folder: Path) -> str:
    book_path = data_folder / "production_raw.xlsx"
    plots_dir, data_dir = _prepare_output_dirs(output_folder)

    processed = pd.read_excel(book_path, sheet_name="processed_hourly")
    processed["timestamp"] = _combine_date_and_hour(processed["date"], processed["hour_start"])
    processed = processed.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")

    expected_hours = pd.date_range(processed["timestamp"].min(), processed["timestamp"].max(), freq="h")
    missing_hours = len(expected_hours.difference(pd.DatetimeIndex(processed["timestamp"])))
    continuity = _longest_hourly_run(processed["timestamp"])

    continuity_frame = pd.DataFrame(
        {
            "start": [processed["timestamp"].min()],
            "end": [processed["timestamp"].max()],
            "observed_hours": [len(processed)],
            "expected_hours": [len(expected_hours)],
            "missing_hours": [missing_hours],
            "longest_run_hours": [continuity["length"]],
        }
    )
    continuity_frame.to_csv(data_dir / "hourly_continuity_summary.csv", index=False)

    daily_summary = pd.read_excel(book_path, sheet_name="daily_operation_summary")
    daily_summary["date"] = pd.to_datetime(daily_summary["date"], errors="coerce").dt.normalize()
    beverage_daily = (
        daily_summary.groupby("date", as_index=False)[
            [
                "production_units",
                "liters_produced",
                "efficiency",
                "gallons_per_hour",
                "monitored_time_dec",
                "operation_time_dec",
                "pause_time_dec",
            ]
        ]
        .sum()
        .sort_values("date")
    )
    beverage_daily.to_csv(data_dir / "beverage_daily_aggregated.csv", index=False)

    daily_diffs = beverage_daily["date"].diff().dropna().dt.days
    daily_gap_counts = daily_diffs.value_counts().sort_index()
    pd.DataFrame({"gap_days": daily_gap_counts.index, "count": daily_gap_counts.values}).to_csv(
        data_dir / "daily_gap_distribution.csv",
        index=False,
    )
    expected_days = pd.date_range(beverage_daily["date"].min(), beverage_daily["date"].max(), freq="D")
    missing_days = len(expected_days.difference(pd.DatetimeIndex(beverage_daily["date"])))

    _plot_beverage_daily_series(
        beverage_daily,
        plots_dir / "daily_production_units.png",
        "production_units",
        "Beverage Daily Production Units",
    )
    _plot_beverage_daily_series(
        beverage_daily,
        plots_dir / "daily_efficiency.png",
        "efficiency",
        "Beverage Daily Efficiency",
    )
    _plot_beverage_daily_time_allocation(
        beverage_daily,
        plots_dir / "daily_time_allocation.png",
    )

    beverage_exploration_markdown = "\n".join(
        [
            "# Beverage Exploration Findings",
            "",
            "## KDD Focus",
            "- The dataset remains useful because it is well documented and was already used in a forecasting study.",
            "- For the current work, the hourly version is being deprioritized because it is too fragmented for a clean continuous-time analysis.",
            "",
            "## Findings",
            f"- The observed hourly series has {len(processed):,} recorded hours across a calendar span that would require {len(expected_hours):,} complete hourly points.",
            f"- That leaves {missing_hours:,} missing hours in the global hourly grid.",
            f"- The longest fully continuous hourly run is only {continuity['length']} hours, from `{continuity['start']}` to `{continuity['end']}`.",
            f"- After daily aggregation there are {len(beverage_daily):,} observed production days across a {len(expected_days):,}-day calendar span, leaving {missing_days:,} missing dates.",
            f"- Daily gaps are irregular as well: {', '.join(f'{int(gap)}d x {int(count)}' for gap, count in daily_gap_counts.items())}.",
            "- Daily aggregation is cleaner than hourly, but it still forms an intermittent production calendar rather than a complete daily time series.",
            "",
            "## Forecasting Ideas",
            "- Drop the hourly version for the main line of analysis because the continuity is too poor for your current objective.",
            "- Keep the daily aggregate as a secondary option if you want day-of-production forecasting rather than continuous calendar forecasting.",
            "- Keep it as a backup because the schema is clean, the context is documented, and it has already supported forecasting research in the associated study.",
            "- If revived later, treat each production day as a short episode or work explicitly with the aggregated daily series.",
        ]
    )
    return beverage_exploration_markdown


DATASET_EXPLORERS = {
    "pt_metro_dataset": explore_metro,
    "garment_sewing_lines_dataset": explore_garment,
    "chinese_boiler_dataset": explore_boiler,
    "beverage_bottling_line_dataset": explore_beverage,
}


def _prepare_output_dirs(output_folder: Path) -> tuple[Path, Path]:
    plots_dir = output_folder / "plots"
    data_dir = output_folder / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, data_dir


def _sample_csv(path: Path, usecols: list[str], sample_step: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
        chunks.append(chunk.iloc[::sample_step].copy())
    return pd.concat(chunks, ignore_index=True)


def _gap_distribution_from_csv(path: Path, timestamp_column: str) -> Counter[float]:
    gap_counts: Counter[float] = Counter()
    previous_timestamp = None
    for chunk in pd.read_csv(path, usecols=[timestamp_column], chunksize=500_000):
        timestamps = pd.to_datetime(chunk[timestamp_column], errors="coerce").dropna()
        if timestamps.empty:
            continue
        if previous_timestamp is not None:
            gap_counts[float((timestamps.iloc[0] - previous_timestamp).total_seconds())] += 1
        gap_counts.update(timestamps.diff().dropna().dt.total_seconds().value_counts().to_dict())
        previous_timestamp = timestamps.iloc[-1]
    return gap_counts


def _plot_gap_histogram(gap_frame: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(gap_frame["gap_seconds"], gap_frame["count"], width=0.8, color="#c44e52")
    axis.set_title(title)
    axis.set_xlabel("Gap in seconds")
    axis.set_ylabel("Occurrences")
    axis.set_xticks(gap_frame["gap_seconds"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_gap_distribution_full(gap_frame: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(gap_frame["gap_seconds"], gap_frame["count"], width=1.0, color="#55a868")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("Gap in seconds (log scale)")
    axis.set_ylabel("Occurrences (log scale)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_family_series(
    frame: pd.DataFrame,
    timestamp_column: str,
    families: dict[str, list[str]],
    plots_dir: Path,
    prefix: str,
) -> None:
    for family_name, columns in families.items():
        fig, axis = plt.subplots(figsize=(14, 5))
        for column in columns:
            axis.plot(frame[timestamp_column], frame[column], linewidth=1.0, label=column)
        axis.set_title(f"{prefix.replace('_', ' ').title()} {family_name.replace('_', ' ').title()}")
        axis.set_xlabel(timestamp_column)
        axis.legend(loc="upper right", fontsize=8, ncol=2 if len(columns) > 4 else 1)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix}_{family_name}.png", dpi=150)
        plt.close(fig)


def _plot_family_heatmaps(
    frame: pd.DataFrame,
    families: dict[str, list[str]],
    plots_dir: Path,
    prefix: str,
) -> None:
    for family_name, columns in families.items():
        if len(columns) < 2:
            continue
        _plot_heatmap(
            frame[columns],
            plots_dir / f"{prefix}_{family_name}_heatmap.png",
            f"{prefix.replace('_', ' ').title()} {family_name.replace('_', ' ').title()} Heatmap",
        )


def _plot_heatmap(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    correlation = frame.corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, cmap="coolwarm", center=0, ax=axis)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_state_heatmap(
    frame: pd.DataFrame,
    timestamp_column: str,
    output_path: Path,
    title: str,
    max_time_points: int,
) -> None:
    numeric = frame.drop(columns=timestamp_column).copy()
    if len(numeric) > max_time_points:
        numeric = numeric.iloc[:max_time_points].copy()
    normalized = (numeric - numeric.mean()) / numeric.std(ddof=0)
    normalized = normalized.T
    fig, axis = plt.subplots(figsize=(14, 8))
    sns.heatmap(normalized, cmap="coolwarm", center=0, ax=axis, cbar_kws={"label": "z-score"})
    axis.set_title(title)
    axis.set_xlabel("Sampled time points")
    axis.set_ylabel("Variables")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _garment_long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        for slot in range(1, 12):
            rows.append(
                {
                    "sequence_id": row_number,
                    "slot": slot,
                    "target": row[f"Target_{slot}"],
                    "output": row[f"Output_{slot}"],
                }
            )
    return pd.DataFrame(rows)


def _plot_garment_slot_profile(frame: pd.DataFrame, output_path: Path) -> None:
    summary = (
        frame.groupby("slot")[["target", "output"]]
        .agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        .reset_index()
    )
    summary.columns = [
        "slot",
        "target_median",
        "target_q25",
        "target_q75",
        "output_median",
        "output_q25",
        "output_q75",
    ]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(summary["slot"], summary["target_median"], color="red", linewidth=2, label="Target median")
    axis.fill_between(summary["slot"], summary["target_q25"], summary["target_q75"], color="red", alpha=0.15)
    axis.plot(summary["slot"], summary["output_median"], color="blue", linewidth=2, label="Output median")
    axis.fill_between(summary["slot"], summary["output_q25"], summary["output_q75"], color="blue", alpha=0.15)
    axis.set_title("Garment Slot Profile: Target vs Output")
    axis.set_xlabel("Slot")
    axis.set_ylabel("Production")
    axis.set_xticks(summary["slot"])
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_garment_sampled_sequences(frame: pd.DataFrame, output_path: Path) -> None:
    sample_ids = sorted(frame["sequence_id"].drop_duplicates().sample(20, random_state=7))
    sampled = frame[frame["sequence_id"].isin(sample_ids)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for sequence_id, group in sampled.groupby("sequence_id"):
        axes[0].plot(group["slot"], group["target"], color="red", alpha=0.25, linewidth=1)
        axes[1].plot(group["slot"], group["output"], color="blue", alpha=0.25, linewidth=1)
    axes[0].set_title("Sampled Target Trajectories")
    axes[1].set_title("Sampled Output Trajectories")
    for axis in axes:
        axis.set_xlabel("Slot")
        axis.set_xticks(sorted(frame["slot"].unique()))
    axes[0].set_ylabel("Production")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_garment_quality_by_slot(frame: pd.DataFrame, output_path: Path) -> None:
    quality = frame[frame["quality_percent"].between(0, 150)].copy()
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=quality, x="slot", y="quality_percent", color="#4c72b0", ax=axis)
    axis.set_title("Garment Quality by Slot")
    axis.set_xlabel("Slot")
    axis.set_ylabel("Quality (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _garment_quality_summary_markdown(frame: pd.DataFrame) -> str:
    quality = frame[frame["quality_percent"].between(0, 150)].copy()
    summary = (
        quality.groupby("slot")["quality_percent"]
        .agg(
            min="min",
            q1=lambda s: s.quantile(0.25),
            median="median",
            q3=lambda s: s.quantile(0.75),
            max="max",
        )
        .reset_index()
    )
    summary = summary.round(2)

    lines = [
        "| Slot | Min (%) | Q1 (%) | Median (%) | Q3 (%) | Max (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.slot)} | {row.min:.2f} | {row.q1:.2f} | {row.median:.2f} | {row.q3:.2f} | {row.max:.2f} |"
        )
    return "\n".join(lines)


def _plot_garment_cluster_profiles(frame: pd.DataFrame, output_path: Path) -> None:
    output_wide = frame.pivot(index="sequence_id", columns="slot", values="output").sort_index()
    scaled = output_wide.div(output_wide.max(axis=1).replace(0, 1), axis=0)
    labels = KMeans(n_clusters=3, random_state=7, n_init=10).fit_predict(scaled)
    output_wide = output_wide.assign(cluster=labels)
    target_wide = frame.pivot(index="sequence_id", columns="slot", values="target").sort_index()
    target_wide = target_wide.assign(cluster=labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for cluster_id, axis in enumerate(axes):
        output_profile = output_wide[output_wide["cluster"] == cluster_id].drop(columns="cluster").median()
        target_profile = target_wide[target_wide["cluster"] == cluster_id].drop(columns="cluster").median()
        slots = output_profile.index.astype(int).to_numpy()
        axis.plot(slots, target_profile.to_numpy(dtype=float), color="red", linewidth=2, label="Target median")
        axis.plot(slots, output_profile.to_numpy(dtype=float), color="blue", linewidth=2, label="Output median")
        axis.set_title(f"Cluster {cluster_id + 1}")
        axis.set_xlabel("Slot")
        axis.set_xticks(slots)
    axes[0].set_ylabel("Production")
    axes[0].legend()
    fig.suptitle("Garment Clustered Trajectory Profiles")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_garment_forecast_signal(frame: pd.DataFrame, output_path: Path) -> None:
    output_wide = frame.pivot(index="sequence_id", columns="slot", values="output").sort_index()
    quality_wide = (frame["output"] / frame["target"]).to_frame("quality")
    quality_wide["sequence_id"] = frame["sequence_id"].values
    quality_wide["slot"] = frame["slot"].values
    quality_wide = quality_wide.pivot(index="sequence_id", columns="slot", values="quality").sort_index()

    output_corr = output_wide.corrwith(output_wide[11])
    quality_corr = quality_wide.corrwith(quality_wide[11])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(output_corr.index, output_corr.values, marker="o", color="#4c72b0")
    axes[0].set_title("Output Slot Correlation with Final Output")
    axes[0].set_xlabel("Slot")
    axes[0].set_ylabel("Correlation")
    axes[0].set_xticks(output_corr.index)
    axes[0].set_ylim(0, 1.05)

    axes[1].scatter(output_wide[5], output_wide[11], s=10, alpha=0.35, color="#dd8452")
    axes[1].set_title("Slot 5 Output vs Final Output")
    axes[1].set_xlabel("Output at Slot 5")
    axes[1].set_ylabel("Final Output")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_beverage_daily_series(
    frame: pd.DataFrame,
    output_path: Path,
    value_column: str,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(frame["date"], frame[value_column], marker="o", linewidth=1.5, color="#4c72b0")
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(value_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_beverage_daily_time_allocation(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(frame["date"], frame["monitored_time_dec"], label="Monitored", linewidth=1.5)
    axis.plot(frame["date"], frame["operation_time_dec"], label="Operation", linewidth=1.5)
    axis.plot(frame["date"], frame["pause_time_dec"], label="Pause", linewidth=1.5)
    axis.set_title("Beverage Daily Time Allocation")
    axis.set_xlabel("Date")
    axis.set_ylabel("Hours")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _combine_date_and_hour(date_column: pd.Series, hour_column: pd.Series) -> pd.Series:
    date_text = pd.to_datetime(date_column, errors="coerce").dt.strftime("%Y-%m-%d")
    hour_text = hour_column.astype("Int64").astype(str).str.zfill(2)
    return pd.to_datetime(date_text + " " + hour_text + ":00:00", errors="coerce")


def _longest_hourly_run(timestamps: pd.Series) -> dict[str, object]:
    ordered = pd.Series(sorted(pd.Series(timestamps).dropna().unique()))
    if ordered.empty:
        return {"start": None, "end": None, "length": 0}

    best_start = current_start = ordered.iloc[0]
    best_length = current_length = 1

    for idx in range(1, len(ordered)):
        if (ordered.iloc[idx] - ordered.iloc[idx - 1]).total_seconds() == 3600:
            current_length += 1
        else:
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
            current_start = ordered.iloc[idx]
            current_length = 1

    if current_length > best_length:
        best_length = current_length
        best_start = current_start

    best_end = best_start + pd.Timedelta(hours=best_length - 1)
    return {"start": best_start, "end": best_end, "length": best_length}


def _remove_files(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
