from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_scripts"))

from forecasting_univariate import (  # noqa: E402
    arima_orders_for_granularity,
    build_training_history_frame,
    build_training_summary_frame,
    build_generic_forecast_metric_row,
    build_split_first_granularity_frame,
    learning_rates_for_candidate,
    inverse_prepared_predictions,
    select_hidden_layer_sizes,
)
from plots import write_mlp_training_diagnostic_plots, write_univariate_metric_comparison  # noqa: E402
from reports import match_selected_candidate_rows  # noqa: E402


class ForecastingUnivariateTests(unittest.TestCase):
    def test_split_first_resampling_keeps_global_bucket_phase(self) -> None:
        dates = pd.date_range("2022-01-01 00:00:00", periods=20, freq="5s")
        raw = pd.DataFrame(
            {
                "unique_id": "subset_B_raw",
                "ds": dates,
                "y": range(20),
                "y_scaled": range(20),
            }
        )
        raw_train = raw.iloc[:10].copy()
        raw_test = raw.iloc[10:].copy()

        frame = build_split_first_granularity_frame(
            raw_train_frame=raw_train,
            raw_test_frame=raw_test,
            id_key="subset_B_30s",
            granularity="30s",
            target_column="target",
            timestamp_column="date",
            granularity_options={"raw": None, "30s": "30s"},
            resampling_policy={
                "timestamp_column": "date",
                "input_frequency": "5s",
                "label": "right",
                "closed": "left",
                "origin": "start",
                "drop_partial_windows": True,
                "default_aggregation": "mean",
                "column_aggregations": {},
            },
        )

        test_dates = frame.loc[frame["split"] == "test", "ds"].reset_index(drop=True)
        self.assertEqual(pd.Timestamp("2022-01-01 00:01:30"), test_dates.iloc[0])
        self.assertTrue(all(date.second in {0, 30} for date in test_dates))

    def test_inverse_first_difference_uses_scaled_previous_value(self) -> None:
        result = inverse_prepared_predictions(
            predicted_model_values=pd.Series([0.5]),
            split_windows={
                "x": pd.DataFrame(),
                "y_model": pd.Series(dtype=float),
                "target_dates": pd.Series(pd.to_datetime(["2022-01-01"])),
                "y_actual": pd.Series([15.0]),
                "prev_y_1": pd.Series([14.0]),
                "prev_y_2": pd.Series([12.0]),
            },
            scaler_row={"scaler": "standard", "train_mean": 10.0, "train_std": 2.0},
            difference_order=1,
        )

        self.assertEqual([15.0], result.to_list())

    def test_inverse_second_difference_uses_scaled_previous_values(self) -> None:
        result = inverse_prepared_predictions(
            predicted_model_values=pd.Series([0.25]),
            split_windows={
                "x": pd.DataFrame(),
                "y_model": pd.Series(dtype=float),
                "target_dates": pd.Series(pd.to_datetime(["2022-01-01"])),
                "y_actual": pd.Series([16.5]),
                "prev_y_1": pd.Series([14.0]),
                "prev_y_2": pd.Series([12.0]),
            },
            scaler_row={"scaler": "standard", "train_mean": 10.0, "train_std": 2.0},
            difference_order=2,
        )

        self.assertEqual([16.5], result.to_list())

    def test_training_history_and_summary_frames_are_built_from_loss_trajectory(self) -> None:
        model = type("Model", (), {"train_trajectories": [(0, 2.0), (1, 1.5), (2, 1.75)]})()

        history = build_training_history_frame(model)
        history["granularity"] = "raw"
        history["candidate_label"] = "raw_d0_smooth_none"
        history["learning_rate_init"] = 0.001

        summary = build_training_summary_frame(
            history,
            pd.DataFrame(
                [
                    {
                        "granularity": "raw",
                        "candidate_label": "raw_d0_smooth_none",
                        "learning_rate_init": 0.001,
                        "mae": 0.5,
                    }
                ]
            ),
        )

        self.assertEqual([2.0, 1.5, 1.5], history["best_train_loss"].to_list())
        self.assertEqual(2, int(summary.loc[0, "training_steps"]))
        self.assertEqual(1.75, float(summary.loc[0, "final_train_loss"]))
        self.assertEqual(0.5, float(summary.loc[0, "mae"]))

    def test_training_diagnostic_plots_are_written(self) -> None:
        history = pd.DataFrame(
            {
                "granularity": ["raw", "raw", "raw"],
                "candidate_label": ["raw_d0_smooth_none"] * 3,
                "learning_rate_init": [0.001, 0.001, 0.001],
                "num_layers": [2, 2, 2],
                "step": [0, 1, 2],
                "train_loss": [2.0, 1.5, 1.75],
                "train_loss_rolling": [2.0, 1.75, 1.75],
                "best_train_loss": [2.0, 1.5, 1.5],
            }
        )
        metrics = pd.DataFrame(
            {
                "granularity": ["raw"],
                "candidate_label": ["raw_d0_smooth_none"],
                "learning_rate_init": [0.001],
                "mae": [0.5],
                "rmse": [0.7],
                "r2": [0.2],
            }
        )

        output_dir = Path.cwd() / ".tmp" / "test_training_diagnostic_plots"
        write_mlp_training_diagnostic_plots(history, metrics, output_dir)

        self.assertTrue((output_dir / "training_summary.png").exists())
        self.assertTrue(
            (
                output_dir
                / "loss_curves"
                / "raw"
                / "raw_d0_smooth_none_lr_0p001_loss_curve.png"
            ).exists()
        )
        self.assertTrue(
            (
                output_dir
                / "learning_curves"
                / "raw"
                / "raw_d0_smooth_none_lr_0p001_learning_curve.png"
            ).exists()
        )
        self.assertTrue(
            (
                output_dir
                / "convergence_by_candidate"
                / "raw"
                / "raw_d0_smooth_none_convergence.png"
            ).exists()
        )

    def test_neuralforecast_hidden_layer_metadata_repeats_width_per_layer(self) -> None:
        self.assertEqual(
            (120, 120, 120),
            select_hidden_layer_sizes(
                120,
                {
                    "hidden_units_strategy": "match_lookback",
                    "num_layers": 3,
                },
            ),
        )

    def test_selected_candidates_do_not_force_learning_rate_when_configured_off(self) -> None:
        self.assertEqual(
            [0.0001, 0.01],
            learning_rates_for_candidate(
                {"selected_learning_rate_init": 0.001},
                [0.0001, 0.01],
                {"use_selected_learning_rates": False},
            ),
        )

    def test_candidate_specific_learning_rate_takes_precedence(self) -> None:
        self.assertEqual(
            [0.01],
            learning_rates_for_candidate(
                {"candidate_label": "30s_d0_smooth_none", "selected_learning_rate_init": 0.0001},
                [0.0001],
                {
                    "use_selected_learning_rates": False,
                    "candidate_settings": {
                        "30s_d0_smooth_none": {
                            "learning_rate_init": 0.01,
                            "min_steps": 5000,
                            "max_steps": 12000,
                        }
                    },
                },
            ),
        )

    def test_report_selected_candidate_matching_treats_learning_rate_as_optional(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "granularity": "raw",
                    "difference_order": 1,
                    "training_smoothing_window": "none",
                    "candidate_label": "raw_d1_smooth_none",
                    "learning_rate_init": 0.01,
                    "mae": 0.2,
                    "rmse": 0.3,
                    "r2": 0.9,
                },
                {
                    "granularity": "raw",
                    "difference_order": 1,
                    "training_smoothing_window": "none",
                    "candidate_label": "raw_d1_smooth_none",
                    "learning_rate_init": 0.0001,
                    "mae": 0.1,
                    "rmse": 0.2,
                    "r2": 0.95,
                },
            ]
        )

        selected = match_selected_candidate_rows(
            comparison,
            [
                {
                    "granularity": "raw",
                    "difference_order": 1,
                    "training_smoothing_window": "none",
                    "role": "best_raw",
                }
            ],
        )

        self.assertEqual(0.0001, float(selected.loc[0, "learning_rate_init"]))
        self.assertEqual("best_raw", selected.loc[0, "role"])

    def test_arima_grid_uses_explicit_equal_p_q_values(self) -> None:
        orders = arima_orders_for_granularity(
            {"p_q_values_by_granularity": {"raw": [30, 60]}, "d_grid": [0, 1, 2]},
            "raw",
            120,
        )

        self.assertNotIn((120, 0, 120), orders)
        self.assertIn((30, 2, 30), orders)
        self.assertEqual(6, len(orders))
        self.assertTrue(all(p == q for p, _d, q in orders))

    def test_arima_grid_can_use_granularity_specific_d_values(self) -> None:
        orders = arima_orders_for_granularity(
            {
                "p_q_values_by_granularity": {"30s": [10, 20, 30]},
                "d_values_by_granularity": {"30s": [0, 1]},
            },
            "30s",
            20,
        )

        self.assertEqual(6, len(orders))
        self.assertEqual({0, 1}, {d for _p, d, _q in orders})
        self.assertTrue(all(p == q for p, _d, q in orders))

    def test_generic_forecast_metric_row_does_not_require_arima_order_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "id_key": ["subset_B_raw", "subset_B_raw"],
                "granularity": ["raw", "raw"],
                "candidate_label": ["raw_ets_ses", "raw_ets_ses"],
                "model": ["ses_level", "ses_level"],
                "train_rows": [10, 10],
                "test_rows": [2, 2],
                "y": [1.0, 2.0],
                "forecast": [1.0, 3.0],
            }
        )

        row = build_generic_forecast_metric_row(frame)

        self.assertEqual("ses_level", row["model"])
        self.assertEqual("raw_ets_ses", row["candidate_label"])
        self.assertEqual(0.5, row["mae"])

    def test_univariate_metric_comparison_writes_r2_plot(self) -> None:
        metrics = pd.DataFrame(
            {
                "candidate_label": ["raw_d1_smooth_none", "raw_d1_smooth_none"],
                "model_family": ["MLP", "ARIMA"],
                "model_variant": ["single", "p120_d1_q120"],
                "r2": [0.9, 0.8],
                "mae": [1.0, 2.0],
                "rmse": [1.5, 2.5],
            }
        )
        output_path = Path.cwd() / ".tmp" / "test_univariate_metric_comparison" / "r2.png"

        write_univariate_metric_comparison(metrics, output_path)

        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
