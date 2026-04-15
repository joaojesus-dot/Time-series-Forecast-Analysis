# Boiler Exploration Findings

## KDD Focus
- Selected because the timestamp cadence is regular and the process diagram gives direct physical context.
- Preprocessing repaired the original dataset with `data_AutoReg.csv` to remove the 30 missing values in `YJJWSLL.AV_0#`.
- Transformation grouped variables into pressure, temperature, flow, oxygen, fan-condition, differential-pressure, and valve-control families.

## Findings
- The repaired dataset was saved to `boiler_repaired.csv` for downstream use.
- Timestamp spacing is constant across the full series: the only observed gap is 5 seconds.
- The variable families align well with the boiler diagram, so this dataset supports both statistical and physical interpretation.
- Family-specific heatmaps were generated to support representative-variable selection inside each physical group.

## Forecasting Ideas
- Forecast `TE_8332A.AV_0#` for boiler outlet steam temperature control.
- Forecast `ZZQBCHLL.AV_0#` for compensated main steam flow demand.
- Forecast `YJJWSLL.AV_0#` to study desuperheating water behavior after repairing the gaps.
- Build models at a family level first to identify whether pressure, temperature, or flow variables contribute the strongest predictive signal.