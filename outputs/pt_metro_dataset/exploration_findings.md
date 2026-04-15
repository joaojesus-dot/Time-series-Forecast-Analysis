# Metro Exploration Findings

## KDD Focus
- Selected because it is multivariate, physically meaningful, and rich enough for autocorrelation and state-based forecasting.
- Preprocessing for exploration used the original timestamps, full gap counting, and a 1-in-20 sample for visualization.
- Transformation grouped variables by physical role: pressure, thermal, load, control, and flow.

## Findings
- The dataset has 337 distinct timestamp gaps, so the cadence is not limited to only 3 values.
- The nominal cadence is still about 10 seconds, with 9s to 13s representing most of the sampling jitter.
- There are 331 gap events above 30 seconds across 319 distinct large-gap sizes, which supports splitting the dataset into continuous segments before autocorrelation work.
- The normalized state heatmap is more informative than pairwise correlation visuals for this dataset because the behavior is dominated by operating modes over time.

## Forecasting Ideas
- Forecast `TP3` or `Reservoirs` to model pneumatic availability and pressure stability.
- Forecast `Motor_current` if the objective is compressor load and operating regime anticipation.
- Forecast `Caudal_impulses` if the focus is downstream air demand rather than compressor internals.
- For the first modeling pass, use continuous segments separated by larger outages and test lag structure inside each segment.