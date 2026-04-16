# Boiler Exploration Findings

## KDD Focus
- The project is now boiler-only, moving from dataset screening to a domain-driven KDD process.
- The repaired working dataset is built from `data.csv` plus the 30 values filled by `data_AutoReg.csv`.
- Variable families are interpreted through the boiler diagram before feature reduction decisions are made.

## Findings
- The repaired dataset was saved to `boiler_repaired.csv` for downstream modeling.
- Timestamp spacing is constant across the full series: the only observed gap is 5 seconds.
- The heatmaps support clear pressure domains, left/right paired sensors, and control-loop structure.
- Raw correlation is not enough for maintenance-oriented interpretation because control action, lag, and operating regime can hide simple physical links.
- Family-level heatmaps are the right tool for representative-variable selection before LSTM experiments on the original and aggregated datasets.

## Dimensionality Reduction Guidance
- Upper furnace pressures A-F form one redundant block; keep one representative for forecasting inputs.
- Steam-side pressures `PTCA_8322A` and `PTCA_8324` form a second redundant pair; keep one representative there as well.
- The oxygen pair is nearly identical, so one variable is enough for the forecasting feature set.
- Left/right paired sensors should not be discarded blindly: one representative supports forecasting, while left/right mismatch can be retained separately for physical diagnostics.