# Garment Exploration Findings

## KDD Focus
- Selected as a slot-sequence production problem rather than a calendar-time problem.
- Preprocessing reshaped the wide table into ordered `(slot, target, output)` observations.
- Transformation added a quality ratio defined as `output / target`.
- In the forecasting plots, `final output` means the last observed slot output, i.e. `Output_11`.

## Findings
- This dataset behaves like repeated within-shift production trajectories.
- Targets define the planned production curve, while outputs describe how each line or shift actually progressed through the slots.
- Treating the slots as an ordered sequence is reasonable for exploratory forecasting, even without an explicit timestamp column.
- Quality values above 100% exist, which means some outputs exceeded the planned target and should not be clipped away in modeling.
- Slot-profile and per-slot quality plots are more interpretable here than a single flattened sequence plot.
- Early-slot outputs already carry strong signal for final output, which supports a forecast-from-partial-shift framing.
- The trajectory clusters separate low, medium, and high production patterns, which is useful for scenario-based forecasting.

## Forecasting Ideas
- Forecast the next slot output from the early slots.
- Forecast the final output level from the first few slots of a sequence.
- Forecast the final quality ratio or final target shortfall to support intervention during the shift.

## Quality Five-Number Summary By Slot

| Slot | Min (%) | Q1 (%) | Median (%) | Q3 (%) | Max (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.42 | 60.82 | 81.13 | 102.92 | 150.00 |
| 2 | 2.00 | 78.47 | 93.90 | 107.40 | 150.00 |
| 3 | 2.55 | 79.29 | 92.68 | 102.96 | 147.55 |
| 4 | 5.76 | 80.20 | 92.99 | 100.32 | 148.43 |
| 5 | 6.47 | 89.57 | 101.40 | 108.95 | 149.85 |
| 6 | 8.52 | 86.60 | 97.61 | 104.97 | 149.60 |
| 7 | 8.22 | 86.18 | 96.61 | 102.88 | 148.82 |
| 8 | 9.63 | 87.50 | 97.61 | 103.60 | 147.09 |
| 9 | 9.07 | 86.91 | 95.91 | 102.01 | 149.25 |
| 10 | 0.90 | 86.65 | 96.31 | 101.87 | 149.95 |
| 11 | 0.65 | 87.43 | 97.39 | 102.50 | 149.95 |