# Boiler Outlier Analysis

## Key Finding
- Outliers are not treated as bad data by default. The strongest events are clustered and multivariate, which is consistent with real operating transitions or control action.

## Main Variables
- `AIR_8301A.AV_0#`: total `2970`, high `1909`, low `1061`.
- `PTCA_8322A.AV_0#`: total `2600`, high `370`, low `2230`.
- `YFJ3_ZD2.AV_0#`: total `2175`, high `2175`, low `0`.
- `YFJ3_AI.AV_0#`: total `1951`, high `1159`, low `792`.
- `ZZQBCHLL.AV_0#`: total `1710`, high `258`, low `1452`.
- `TE_8313B.AV_0#`: total `1247`, high `181`, low `1066`.

## Highest-Density Windows
- `2022-03-31 16:30:00`: `261` events across `6` variables.
- `2022-03-31 16:35:00`: `216` events across `7` variables.
- `2022-03-31 16:25:00`: `197` events across `6` variables.
- `2022-03-30 07:45:00`: `195` events across `4` variables.
- `2022-03-31 11:55:00`: `165` events across `4` variables.

## Strongest Simultaneous Events
- `2022-03-31 16:33:54`: `6` simultaneous variables: `TE_8303.AV_0#; TE_8313B.AV_0#; TE_8319A.AV_0#; YFJ3_AI.AV_0#; YFJ3_ZD2.AV_0#; ZZQBCHLL.AV_0#`.
- `2022-03-31 16:33:29`: `6` simultaneous variables: `TE_8303.AV_0#; TE_8313B.AV_0#; TE_8319A.AV_0#; YFJ3_AI.AV_0#; YFJ3_ZD2.AV_0#; ZZQBCHLL.AV_0#`.
- `2022-03-31 16:34:44`: `6` simultaneous variables: `TE_8303.AV_0#; TE_8313B.AV_0#; TE_8319A.AV_0#; YFJ3_AI.AV_0#; YFJ3_ZD2.AV_0#; ZZQBCHLL.AV_0#`.
- `2022-03-31 16:35:29`: `6` simultaneous variables: `TE_8303.AV_0#; TE_8313B.AV_0#; TE_8319A.AV_0#; YFJ3_AI.AV_0#; YFJ3_ZD2.AV_0#; ZZQBCHLL.AV_0#`.
- `2022-03-31 16:34:29`: `6` simultaneous variables: `TE_8303.AV_0#; TE_8313B.AV_0#; TE_8319A.AV_0#; YFJ3_AI.AV_0#; YFJ3_ZD2.AV_0#; ZZQBCHLL.AV_0#`.

## Modeling Decision
- Do not delete outliers automatically.
- Keep multivariate clusters as real operating information.
- Review isolated single-variable spikes separately if they affect model training.