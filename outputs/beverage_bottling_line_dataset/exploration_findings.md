# Beverage Exploration Findings

## KDD Focus
- The dataset remains useful because it is well documented and was already used in a forecasting study.
- For the current work, the hourly version is being deprioritized because it is too fragmented for a clean continuous-time analysis.

## Findings
- The observed hourly series has 272 recorded hours across a calendar span that would require 4,877 complete hourly points.
- That leaves 4,605 missing hours in the global hourly grid.
- The longest fully continuous hourly run is only 10 hours, from `2022-07-22 11:00:00` to `2022-07-22 20:00:00`.
- After daily aggregation there are 56 observed production days across a 204-day calendar span, leaving 148 missing dates.
- Daily gaps are irregular as well: 1d x 12, 2d x 8, 3d x 16, 4d x 7, 5d x 4, 6d x 3, 7d x 2, 9d x 2, 29d x 1.
- Daily aggregation is cleaner than hourly, but it still forms an intermittent production calendar rather than a complete daily time series.

## Forecasting Ideas
- Drop the hourly version for the main line of analysis because the continuity is too poor for your current objective.
- Keep the daily aggregate as a secondary option if you want day-of-production forecasting rather than continuous calendar forecasting.
- Keep it as a backup because the schema is clean, the context is documented, and it has already supported forecasting research in the associated study.
- If revived later, treat each production day as a short episode or work explicitly with the aggregated daily series.