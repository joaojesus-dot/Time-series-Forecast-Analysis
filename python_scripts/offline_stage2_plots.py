from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

NUMERIC_COLUMNS = [
    "mae",
    "rmse",
    "mape",
    "smape",
    "bias",
    "r2",
    "training_seconds",
    "learning_rate_init",
    "num_layers",
    "hidden_units",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate Stage 2 result plots from saved CSV artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chinese_boiler_dataset"))
    parser.add_argument(
        "--stage1-run",
        type=Path,
        default=Path(
            "outputs/chinese_boiler_dataset/history/"
            "20260430T014551_standard-real-mlp-1step-review_dc82a37d"
        ),
    )
    args = parser.parse_args()

    dataset_output = args.output_dir
    tables_dir = dataset_output / "tables" / "forecasting_summary"
    plot_dir = dataset_output / "plots" / "forecasting" / "univariate" / "stage2_results"
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    stage2 = load_stage2_metrics(tables_dir)
    stage1 = load_stage1_baseline(args.stage1_run)
    training_summary = read_csv(tables_dir / "univariate_mlp_training_summary.csv")
    training_history = load_training_history(dataset_output)
    forecasts = load_stage2_forecasts(dataset_output)

    model_results = plot_dir / "model_results"
    write_stage2_leaderboards(stage2, plot_dir / "all_model_comparison" / "leaderboards")
    write_best_family_comparison(stage2, plot_dir / "all_model_comparison" / "best_family_comparison")
    write_family_clustered_comparisons(stage2, plot_dir / "all_model_comparison" / "clustered_by_family")
    write_mlp_architecture_effects(stage2, model_results / "mlp" / "architecture")
    write_runtime_accuracy_plots(stage2, training_summary, model_results / "mlp" / "runtime_accuracy")
    write_training_convergence_plots(training_history, stage2, model_results / "mlp" / "training_convergence")
    write_arima_plots(stage2, model_results / "arima")
    write_exponential_smoothing_plots(stage2, model_results / "exponential_smoothing")
    write_forecast_result_plots(forecasts, stage2, plot_dir / "forecast_results")
    write_stage1_stage2_comparisons(stage1, stage2, plot_dir / "baseline_comparison")

    print(f"Wrote offline Stage 2 plots to {plot_dir}", flush=True)
    return 0


def load_stage2_metrics(tables_dir: Path) -> pd.DataFrame:
    frames = []
    mlp = read_csv(tables_dir / "univariate_mlp_test_comparison.csv")
    if not mlp.empty:
        frames.append(
            mlp.assign(
                model_family="MLP",
                family_model=mlp["model_variant"],
                preprocessing_candidate=mlp["candidate_label"],
                stage="stage2_3min",
            )
        )

    arima = read_csv(tables_dir / "univariate_arima_metrics.csv")
    if not arima.empty:
        frames.append(
            arima.assign(
                model_family="ARIMA",
                family_model=arima["model"],
                preprocessing_candidate=arima["granularity"] + "_arima",
                stage="stage2_3min",
            )
        )

    ets = read_csv(tables_dir / "univariate_exponential_smoothing_metrics.csv")
    if not ets.empty:
        frames.append(
            ets.assign(
                model_family="Exponential smoothing",
                family_model=ets["model"],
                preprocessing_candidate=ets["granularity"] + "_ets",
                stage="stage2_3min",
            )
        )

    if not frames:
        return pd.DataFrame()
    return coerce_numeric(pd.concat(frames, ignore_index=True))


def load_stage1_baseline(stage1_run: Path) -> pd.DataFrame:
    path = stage1_run / "tables" / "forecasting_summary" / "univariate_mlp_test_comparison.csv"
    baseline = read_csv(path)
    if baseline.empty:
        return baseline
    baseline = baseline[baseline["split"] == "test"].copy()
    baseline = coerce_numeric(baseline)
    baseline["model_family"] = "Stage 1 baseline MLP"
    baseline["family_model"] = "baseline_1step_mlp"
    baseline["preprocessing_candidate"] = baseline["candidate_label"]
    baseline["stage"] = "stage1_1step"
    return baseline


def load_training_history(dataset_output: Path) -> pd.DataFrame:
    candidates = [
        dataset_output / "forecasting" / "univariate" / "training_results" / "mlp_training_history.csv",
        dataset_output / "forecasting" / "univariate" / "mlp_training_history.csv",
    ]
    for path in candidates:
        if path.exists():
            return coerce_numeric(pd.read_csv(path))
    return pd.DataFrame()


def load_stage2_forecasts(dataset_output: Path) -> pd.DataFrame:
    forecast_dir = dataset_output / "forecasting" / "univariate"
    frames = []
    paths = [
        ("MLP", forecast_dir / "mlp_forecasts.csv"),
        ("ARIMA", forecast_dir / "arima_forecasts.csv"),
        ("Exponential smoothing", forecast_dir / "exponential_smoothing_forecasts.csv"),
    ]
    for family, path in paths:
        frame = read_csv(path)
        if frame.empty:
            continue
        frame["model_family"] = family
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "ds" in combined.columns:
        combined["ds"] = pd.to_datetime(combined["ds"], errors="coerce")
    return coerce_numeric(combined)


def write_stage2_leaderboards(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for granularity, frame in metrics.groupby("granularity"):
        ordered = top_rows_with_required_family(frame, "mae", 18, "Exponential smoothing")
        ordered["plot_label"] = ordered.apply(short_model_label, axis=1)
        height = max(6, 0.42 * len(ordered.index))
        fig, axis = plt.subplots(figsize=(14, height))
        sns.barplot(data=ordered, y="plot_label", x="mae", hue="model_family", dodge=False, ax=axis)
        axis.set_title(f"{granularity} - Stage 2 MAE Leaderboard")
        axis.set_xlabel("MAE")
        axis.set_ylabel("")
        axis.legend(loc="best", fontsize=8, title="Family")
        save_figure(fig, output_dir / f"{granularity}_mae_leaderboard.png")

        r2_ordered = top_rows_with_required_family(frame, "r2", 18, "Exponential smoothing", ascending=False)
        r2_ordered["plot_label"] = r2_ordered.apply(short_model_label, axis=1)
        fig, axis = plt.subplots(figsize=(14, max(6, 0.42 * len(r2_ordered.index))))
        sns.barplot(data=r2_ordered, y="plot_label", x="r2", hue="model_family", dodge=False, ax=axis)
        axis.set_title(f"{granularity} - Stage 2 R2 Leaderboard")
        axis.set_xlabel("R2")
        axis.set_ylabel("")
        axis.legend(loc="best", fontsize=8, title="Family")
        save_figure(fig, output_dir / f"{granularity}_r2_leaderboard.png")


def write_best_family_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    best = best_by(metrics, ["granularity", "model_family"], "mae")
    for metric in ["mae", "rmse", "r2"]:
        fig, axis = plt.subplots(figsize=(12, 6))
        sns.barplot(data=best, x="granularity", y=metric, hue="model_family", ax=axis)
        axis.set_title(f"Best Stage 2 Model Per Family - {metric.upper()}")
        axis.set_xlabel("Granularity")
        axis.set_ylabel(metric.upper())
        axis.legend(loc="best", fontsize=8, title="Family")
        save_figure(fig, output_dir / f"best_family_{metric}.png")

    write_best_table(best, output_dir / "best_family_models.csv")


def write_family_clustered_comparisons(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    best = best_by(metrics, ["granularity", "model_family"], "mae")
    for metric in ["mae", "rmse", "r2"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
        for axis, granularity in zip(axes, sorted(best["granularity"].dropna().unique())):
            frame = best[best["granularity"] == granularity].copy()
            sns.barplot(data=frame, x="model_family", y=metric, ax=axis)
            axis.set_title(f"{granularity}")
            axis.set_xlabel("")
            axis.set_ylabel(metric.upper())
            axis.tick_params(axis="x", rotation=25)
        fig.suptitle(f"Best Model Per Family - {metric.upper()}")
        save_figure(fig, output_dir / f"clustered_best_family_{metric}.png")

    for granularity, frame in metrics.groupby("granularity"):
        best_per_family = best_by(frame, ["model_family"], "mae")
        candidates = []
        for family, family_frame in frame.groupby("model_family"):
            candidates.append(family_frame.sort_values("mae").head(6))
        compact = pd.concat(candidates, ignore_index=True) if candidates else best_per_family
        compact["plot_label"] = compact.apply(short_model_label, axis=1)
        fig, axis = plt.subplots(figsize=(14, max(6, 0.38 * len(compact.index))))
        sns.barplot(data=compact.sort_values(["model_family", "mae"]), y="plot_label", x="mae", hue="model_family", dodge=False, ax=axis)
        axis.set_title(f"{granularity} - Model Results Clustered By Family")
        axis.set_xlabel("MAE")
        axis.set_ylabel("")
        axis.legend(loc="best", fontsize=8, title="Family")
        save_figure(fig, output_dir / f"{granularity}_clustered_by_family_mae.png")

    best.to_csv(output_dir / "best_by_family_metrics.csv", index=False)


def write_arima_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    arima = metrics[metrics["model_family"] == "ARIMA"].copy()
    if arima.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["mae", "rmse", "r2"]:
        fig, axis = plt.subplots(figsize=(13, 6))
        sns.lineplot(data=arima, x="p", y=metric, hue="granularity", style="d", markers=True, dashes=False, ax=axis)
        axis.set_title(f"ARIMA Stage 2 {metric.upper()} By p=q And d")
        axis.set_xlabel("p = q")
        axis.set_ylabel(metric.upper())
        axis.legend(loc="best", fontsize=8, title="Granularity / d")
        save_figure(fig, output_dir / f"arima_{metric}_by_order.png")

    best = best_by(arima, ["granularity"], "mae")
    fig, axis = plt.subplots(figsize=(12, 5))
    sns.barplot(data=best.sort_values("mae"), x="granularity", y="mae", hue="model", dodge=False, ax=axis)
    axis.set_title("Best ARIMA Order Per Granularity")
    axis.set_xlabel("Granularity")
    axis.set_ylabel("MAE")
    axis.legend(loc="best", fontsize=8, title="Order")
    save_figure(fig, output_dir / "best_arima_per_granularity.png")
    arima.to_csv(output_dir / "arima_metrics.csv", index=False)


def write_exponential_smoothing_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    smoothing = metrics[metrics["model_family"] == "Exponential smoothing"].copy()
    if smoothing.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["mae", "rmse", "r2"]:
        fig, axis = plt.subplots(figsize=(12, 6))
        sns.barplot(data=smoothing, x="granularity", y=metric, hue="model", ax=axis)
        axis.set_title(f"Exponential Smoothing Stage 2 {metric.upper()}")
        axis.set_xlabel("Granularity")
        axis.set_ylabel(metric.upper())
        axis.legend(loc="best", fontsize=8, title="Variant")
        save_figure(fig, output_dir / f"exponential_smoothing_{metric}.png")

    best = best_by(smoothing, ["granularity"], "mae")
    fig, axis = plt.subplots(figsize=(12, 5))
    sns.barplot(data=best.sort_values("mae"), x="granularity", y="mae", hue="model", dodge=False, ax=axis)
    axis.set_title("Best Exponential Smoothing Variant Per Granularity")
    axis.set_xlabel("Granularity")
    axis.set_ylabel("MAE")
    axis.legend(loc="best", fontsize=8, title="Variant")
    save_figure(fig, output_dir / "best_exponential_smoothing_per_granularity.png")

    smoothing.to_csv(output_dir / "exponential_smoothing_metrics.csv", index=False)


def write_mlp_architecture_effects(metrics: pd.DataFrame, output_dir: Path) -> None:
    mlp = metrics[metrics["model_family"] == "MLP"].copy()
    if mlp.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    mlp["layer_lr"] = (
        mlp["num_layers"].astype("Int64").astype(str)
        + " layers | lr="
        + mlp["learning_rate_init"].map(lambda value: f"{float(value):g}")
    )

    for granularity, frame in mlp.groupby("granularity"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.pointplot(data=frame, x="num_layers", y="mae", hue="learning_rate_init", dodge=True, errorbar=None, ax=axes[0])
        axes[0].set_title(f"{granularity} - MLP Depth/LR Effect On MAE")
        axes[0].set_xlabel("Hidden layers")
        axes[0].set_ylabel("MAE")
        sns.pointplot(data=frame, x="num_layers", y="r2", hue="learning_rate_init", dodge=True, errorbar=None, ax=axes[1])
        axes[1].set_title(f"{granularity} - MLP Depth/LR Effect On R2")
        axes[1].set_xlabel("Hidden layers")
        axes[1].set_ylabel("R2")
        save_figure(fig, output_dir / f"{granularity}_depth_lr_effect.png")

    best_mlp = best_by(mlp, ["granularity", "candidate_label"], "mae")
    fig, axis = plt.subplots(figsize=(14, max(6, 0.35 * len(best_mlp.index))))
    sns.barplot(data=best_mlp.sort_values("mae"), y="candidate_label", x="mae", hue="granularity", dodge=False, ax=axis)
    axis.set_title("Best MLP Result Per Candidate")
    axis.set_xlabel("MAE")
    axis.set_ylabel("Candidate")
    save_figure(fig, output_dir / "best_mlp_per_candidate_mae.png")


def write_runtime_accuracy_plots(metrics: pd.DataFrame, training_summary: pd.DataFrame, output_dir: Path) -> None:
    mlp = metrics[metrics["model_family"] == "MLP"].copy()
    if mlp.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(data=mlp, x="training_seconds", y="mae", hue="granularity", style="num_layers", ax=axes[0])
    axes[0].set_title("MLP Runtime vs MAE")
    axes[0].set_xlabel("Training seconds")
    axes[0].set_ylabel("MAE")
    sns.scatterplot(data=mlp, x="training_seconds", y="r2", hue="granularity", style="num_layers", ax=axes[1])
    axes[1].set_title("MLP Runtime vs R2")
    axes[1].set_xlabel("Training seconds")
    axes[1].set_ylabel("R2")
    save_figure(fig, output_dir / "mlp_runtime_vs_accuracy.png")

    if not training_summary.empty:
        summary = coerce_numeric(training_summary)
        fig, axis = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=summary, x="best_train_loss", y="mae", hue="granularity", style="learning_rate_init", ax=axis)
        axis.set_title("Best Training Loss vs Test MAE")
        axis.set_xlabel("Best training loss")
        axis.set_ylabel("Test MAE")
        save_figure(fig, output_dir / "best_train_loss_vs_mae.png")


def write_stage1_stage2_comparisons(stage1: pd.DataFrame, stage2: pd.DataFrame, output_dir: Path) -> None:
    if stage1.empty or stage2.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_best_candidate = best_by(stage1, ["granularity", "candidate_label"], "mae")
    stage2_mlp = stage2[stage2["model_family"] == "MLP"].copy()
    stage2_best_candidate = best_by(stage2_mlp, ["granularity", "candidate_label"], "mae")
    matched = stage1_best_candidate.merge(
        stage2_best_candidate,
        on=["granularity", "candidate_label"],
        suffixes=("_stage1", "_stage2"),
    )
    if not matched.empty:
        for metric in ["mae", "rmse", "r2"]:
            plot_stage_metric_delta(matched, metric, output_dir / f"candidate_stage1_vs_stage2_{metric}.png")
        matched.to_csv(output_dir / "candidate_stage1_vs_stage2_metrics.csv", index=False)

    stage1_best_granularity = best_by(stage1, ["granularity"], "mae")
    stage2_best_family = best_by(stage2, ["granularity", "model_family"], "mae")
    family_baseline = stage2_best_family.merge(
        stage1_best_granularity[["granularity", "mae", "rmse", "r2"]],
        on="granularity",
        suffixes=("_stage2", "_stage1_baseline"),
    )
    for metric in ["mae", "rmse", "r2"]:
        plot_family_baseline_comparison(family_baseline, metric, output_dir / f"family_vs_stage1_baseline_{metric}.png")
    family_baseline.to_csv(output_dir / "family_vs_stage1_baseline_metrics.csv", index=False)


def write_forecast_result_plots(forecasts: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    if forecasts.empty or metrics.empty:
        return
    required = {"ds", "y", "forecast", "granularity", "model_family"}
    if not required.issubset(forecasts.columns):
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for family, family_metrics in metrics.groupby("model_family"):
        family_forecasts = forecasts[forecasts["model_family"] == family].copy()
        if family_forecasts.empty:
            continue
        family_dir = output_dir / safe_filename(family.lower())
        best_rows = best_by(family_metrics, ["granularity"], "mae")
        for row in best_rows.to_dict("records"):
            selected = select_matching_forecast_rows(family_forecasts, row).sort_values("ds")
            if selected.empty:
                continue
            write_single_forecast_overlay(
                selected,
                family_dir / f"{row['granularity']}_best_{safe_filename(family)}_forecast.png",
                f"{row['granularity']} - Best {family} Forecast",
            )

    best_family_rows = best_by(metrics, ["granularity", "model_family"], "mae")
    for granularity, family_rows in best_family_rows.groupby("granularity"):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        plotted = False
        output_scale = "original"
        for row in family_rows.to_dict("records"):
            family_forecasts = forecasts[forecasts["model_family"] == row["model_family"]]
            selected = select_matching_forecast_rows(family_forecasts, row).sort_values("ds").head(700)
            if selected.empty:
                continue
            if not plotted:
                axes[0].plot(selected["ds"], selected["y"], color="#222222", linewidth=1.1, label="actual")
                output_scale = first_forecast_output_scale(selected)
                plotted = True
            label = str(row["model_family"])
            axes[0].plot(selected["ds"], selected["forecast"], linewidth=1.0, label=label)
            axes[1].plot(selected["ds"], selected["forecast"] - selected["y"], linewidth=1.0, label=label)
        if not plotted:
            plt.close(fig)
            continue
        y_label = "Scaled target" if output_scale == "scaled" else "Target"
        axes[0].set_title(f"{granularity} - Best Forecast By Family")
        axes[0].set_ylabel(y_label)
        axes[0].legend(loc="best", fontsize=8)
        axes[1].axhline(0, color="#333333", linewidth=0.8)
        axes[1].set_title("Forecast Error")
        axes[1].set_xlabel("date")
        axes[1].set_ylabel(f"Forecast - actual ({output_scale})")
        axes[1].legend(loc="best", fontsize=8)
        save_figure(fig, output_dir / "best_family_overlays" / f"{granularity}_best_family_forecasts.png")


def select_matching_forecast_rows(forecasts: pd.DataFrame, metric_row: dict[str, object]) -> pd.DataFrame:
    selected = forecasts[forecasts["granularity"] == metric_row["granularity"]].copy()
    for column in ["candidate_label", "model", "model_variant", "learning_rate_init", "p", "d", "q"]:
        if column not in selected.columns or column not in metric_row or pd.isna(metric_row[column]):
            continue
        selected = selected[selected[column].astype(str) == str(metric_row[column])]
    return selected


def write_single_forecast_overlay(frame: pd.DataFrame, output_path: Path, title: str, max_points: int = 700) -> None:
    plot_frame = frame.head(max_points).copy()
    if plot_frame.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    output_scale = first_forecast_output_scale(plot_frame)
    y_label = "Scaled target" if output_scale == "scaled" else "Target"
    axes[0].plot(plot_frame["ds"], plot_frame["y"], color="#222222", linewidth=1.1, label="actual")
    axes[0].plot(plot_frame["ds"], plot_frame["forecast"], color="#1f4b99", linewidth=1.0, label="forecast")
    axes[0].set_title(title)
    axes[0].set_ylabel(y_label)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(plot_frame["ds"], plot_frame["forecast"] - plot_frame["y"], color="#b24d2b", linewidth=1.0)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Forecast Error")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel(f"Forecast - actual ({output_scale})")
    save_figure(fig, output_path)


def first_forecast_output_scale(frame: pd.DataFrame) -> str:
    if "forecast_output_scale" not in frame.columns:
        return "original"
    values = frame["forecast_output_scale"].dropna()
    if values.empty:
        return "original"
    return str(values.iloc[0])


def write_training_convergence_plots(training_history: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    if training_history.empty:
        return
    required = {"granularity", "candidate_label", "learning_rate_init", "step", "train_loss"}
    if not required.issubset(training_history.columns):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    history = training_history.copy()
    history["loss_for_plot"] = history.get("train_loss_rolling", history["train_loss"])
    if "num_layers" not in history.columns and "model_variant" in history.columns:
        history["num_layers"] = history["model_variant"].str.extract(r"_(\d+)_hidden").astype(float)
    history["curve_label"] = (
        history["num_layers"].astype("Int64").astype(str)
        + "L lr="
        + history["learning_rate_init"].map(lambda value: f"{float(value):g}")
    )

    best_candidates = []
    mlp = metrics[metrics["model_family"] == "MLP"].copy()
    if not mlp.empty:
        best_candidates = best_by(mlp, ["granularity"], "mae")["candidate_label"].to_list()
    selected_history = history[history["candidate_label"].isin(best_candidates)].copy()
    if selected_history.empty:
        selected_history = history.copy()

    for (granularity, candidate_label), frame in selected_history.groupby(["granularity", "candidate_label"]):
        sampled = sample_curves(frame, ["curve_label"], max_points=900)
        if sampled.empty:
            continue
        fig, axis = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=sampled, x="step", y="loss_for_plot", hue="curve_label", ax=axis, linewidth=1.1)
        axis.set_title(f"{granularity} - {candidate_label} Training Convergence")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Rolling train loss")
        axis.legend(loc="best", fontsize=8)
        save_figure(fig, output_dir / f"{granularity}_{safe_filename(candidate_label)}_convergence.png")


def plot_stage_metric_delta(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_frame = frame[["granularity", "candidate_label", f"{metric}_stage1", f"{metric}_stage2"]].copy()
    long = plot_frame.melt(
        id_vars=["granularity", "candidate_label"],
        value_vars=[f"{metric}_stage1", f"{metric}_stage2"],
        var_name="stage",
        value_name=metric,
    )
    long["stage"] = long["stage"].map({f"{metric}_stage1": "Stage 1 baseline", f"{metric}_stage2": "Stage 2 MLP"})
    fig, axis = plt.subplots(figsize=(14, max(6, 0.4 * long["candidate_label"].nunique())))
    sns.barplot(data=long, y="candidate_label", x=metric, hue="stage", ax=axis)
    axis.set_title(f"Candidate {metric.upper()} Comparison: Stage 1 Baseline vs Stage 2 MLP")
    axis.set_xlabel(metric.upper())
    axis.set_ylabel("Candidate")
    save_figure(fig, output_path)


def plot_family_baseline_comparison(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    rows = []
    for row in frame.to_dict("records"):
        rows.append({"granularity": row["granularity"], "model_family": row["model_family"], metric: row[f"{metric}_stage2"]})
        rows.append({"granularity": row["granularity"], "model_family": "Stage 1 baseline", metric: row[f"{metric}_stage1_baseline"]})
    plot_frame = pd.DataFrame(rows).drop_duplicates()
    fig, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_frame, x="granularity", y=metric, hue="model_family", ax=axis)
    axis.set_title(f"{metric.upper()} Comparison By Family Against Stage 1 Baseline")
    axis.set_xlabel("Granularity")
    axis.set_ylabel(metric.upper())
    axis.legend(loc="best", fontsize=8, title="Family")
    save_figure(fig, output_path)


def best_by(frame: pd.DataFrame, group_columns: list[str], metric: str) -> pd.DataFrame:
    ascending = metric != "r2"
    return (
        frame.sort_values(group_columns + [metric, "rmse"], ascending=[True] * len(group_columns) + [ascending, True])
        .groupby(group_columns, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def top_rows_with_required_family(
    frame: pd.DataFrame,
    metric: str,
    limit: int,
    required_family: str,
    ascending: bool = True,
) -> pd.DataFrame:
    ranked = frame.sort_values(metric, ascending=ascending).head(limit).copy()
    required = frame[frame["model_family"] == required_family].copy()
    if required.empty:
        return ranked
    combined = pd.concat([ranked, required], ignore_index=True)
    dedupe_columns = [
        column
        for column in ["model_family", "model", "candidate_label", "granularity", "learning_rate_init", "num_layers", "p", "d", "q"]
        if column in combined.columns
    ]
    return combined.drop_duplicates(subset=dedupe_columns).sort_values(metric, ascending=ascending).reset_index(drop=True)


def short_model_label(row: pd.Series) -> str:
    if row["model_family"] == "MLP":
        return (
            f"{row['candidate_label']} | {int(row['num_layers'])}L "
            f"lr={float(row['learning_rate_init']):g}"
        )
    if row["model_family"] == "Exponential smoothing":
        return f"Exp smoothing | {row['model']}"
    return f"{row['model_family']} | {row['model']}"


def write_best_table(frame: pd.DataFrame, output_path: Path) -> None:
    columns = [
        column
        for column in [
            "granularity",
            "model_family",
            "model",
            "candidate_label",
            "mae",
            "rmse",
            "r2",
            "training_seconds",
        ]
        if column in frame.columns
    ]
    frame[columns].to_csv(output_path, index=False)


def sample_curves(frame: pd.DataFrame, group_columns: list[str], max_points: int) -> pd.DataFrame:
    sampled = []
    for _values, group in frame.groupby(group_columns):
        ordered = group.sort_values("step")
        if len(ordered.index) <= max_points:
            sampled.append(ordered)
            continue
        stride = max(1, len(ordered.index) // max_points)
        chunk = ordered.iloc[::stride].copy()
        if chunk.iloc[-1]["step"] != ordered.iloc[-1]["step"]:
            chunk = pd.concat([chunk, ordered.tail(1)], ignore_index=True)
        sampled.append(chunk)
    return pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def safe_filename(value: str) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "_")
        .replace("=", "_")
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
