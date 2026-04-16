# Boiler Physical Analysis

This document establishes the first-principles interpretation of the boiler before the KDD modeling stage. The correlation heatmaps are treated as supporting evidence rather than as the source of causality.

## 1. Physical Sensor Network
- `PT_8313A` to `PT_8313F` are treated as one upper-furnace pressure domain observed at different positions.
- `PTCA_8322A` and `PTCA_8324` are treated as the steam/container pressure pair.
- `TE_8319A/B`, `TE_8313B`, `TE_8303`, `TE_8304`, and `TE_8332A` are treated as the main thermal chain: economizer/flue gas, furnace, air-preheater outlet, and outlet steam temperature.
- `FT_8301`, `FT_8302`, `FT_8306A/B`, `YJJWSLL`, and `ZZQBCHLL` are treated as the main flow path variables: air/return-air flow, spray-water flow, and compensated steam flow.
- `AIR_8301A` and `AIR_8301B` are treated as the left/right oxygen pair.
- `YFJ3_AI`, `YFJ3_ZD1`, and `YFJ3_ZD2` are treated as fan-condition variables: current plus vibration.
- `SXLTCYZ`, `SXLTCYY`, `ZCLCCY`, and `YCLCCY` are treated as the differential-pressure field for gas-path resistance and imbalance.
- `TV_8329ZC` together with `YJJWSLL` is treated as the desuperheating control loop.

## 2. Expected Causal Links
- Higher primary and secondary air flow is expected to increase O2 unless fuel/feed rises proportionally.
- Excess air is expected to reduce thermal efficiency and lower furnace temperature through dilution.
- Economizer and air-preheater outlet temperatures are expected to reflect both thermal driving force and flow rate.
- Steam-production load is expected to couple to furnace temperature, gas-path behavior, and draft effort.
- Desuperheater valve action and spray-water flow are expected to reduce outlet steam temperature.
- Differential pressures are expected to rise with gas or air flow, while abnormal increases at similar flow suggest fouling or blockage.
- Induced-draft fan current is expected to rise with draft duty, gas flow, and path resistance.
- Left/right duplicate sensors are expected to move together; persistent asymmetry is physically meaningful.
- Pressure and flow are expected to react faster than temperatures, so lag must be considered.

## 3. Prior Thermodynamic Dependency Graph
- `FT_8301/FT_8302/FT_8306A/FT_8306B` -> `AIR_8301A/B`, `PT_8313*`, `SXLTCYZ/SXLTCYY/ZCLCCY/YCLCCY`.
- `TE_8313B` + gas-path state -> `TE_8319A/B`.
- flue-gas thermal state -> `TE_8303/TE_8304`.
- `ZZQBCHLL` + `TV_8329ZC/YJJWSLL` -> `TE_8332A`.
- `SXLTCYZ/SXLTCYY/ZCLCCY/YCLCCY` -> `YFJ3_AI`.
- mechanical condition -> `YFJ3_ZD1/YFJ3_ZD2`.

## 4. Sensor-to-Sensor Relationship Table
- The structured sensor-to-sensor relationship table is stored separately in `outputs/chinese_boiler_dataset/data/sensor_relationship_table.csv`.
- That CSV is the operational reference because it is easier to filter, sort, and extend than a markdown table.

## Physical Findings
- The heatmaps support clear pressure domains: the upper furnace pressure block and the steam-side pressure pair are both physically coherent.
- The heatmaps support clear left/right paired sensors in the economizer temperatures, oxygen pair, and layer differential pressures.
- The desuperheating variables behave like a control loop, so their relationships should be interpreted through control action rather than simple raw correlation.
- The gas-path, steam-load, and thermal variables are coupled, but not in a simple static way. Lag, operating regime, and control compensation matter.
- The weaker correlations around outlet steam temperature, O2 versus air flow, and return-air behavior do not invalidate the thermodynamic picture; they indicate that the plant is under active control and is not behaving like an open-loop system.

## Representative-Variable View
- The upper furnace pressure sensors can be reduced to one representative, and the steam-side pressure pair can be reduced to one representative, without losing the main pressure-domain structure.
- The full temperature family should not be collapsed to one variable. A representative economizer outlet sensor, an air-preheater outlet temperature, and the target-side temperature view should be preserved.
- The oxygen pair can be reduced to one representative for forecasting inputs.
- Fan current plus one vibration channel should be retained when maintenance-aware physical context is needed instead of pure compression.
- Left/right asymmetry should be preserved as a diagnostic concept even when the forecasting feature set keeps only one representative per pair.