# Boiler Feature Selection Note

## Objective
The forecasting target is `TE_8313B.AV_0#`, the temperature in the upper part of the furnace chamber. The feature-selection process is designed to reduce dimensionality while preserving the main physical domains of the boiler: pressure, temperature, flow, oxygen, fan condition, differential pressure, and control action.

The selection is not based only on raw correlation. Boiler operation is controlled, and control loops can hide simple thermodynamic relationships. Therefore, the feature sets are defined in stages: first from physical structure, then from heatmap evidence, and finally from explicit control-loop information.

## Selection Principles
- Redundant sensors are compressed when they observe the same physical domain.
- At least one representative is preserved for each relevant boiler segment.
- Left/right pairs are reduced for forecasting, but asymmetry remains important for maintenance interpretation.
- Control-loop variables are separated from the base physical set because they may improve forecasting while changing the interpretation of the model.

## Candidate A: Physical-Only Reduced Dataset
Candidate A is the dataset that would be selected using only the physical analysis and the boiler diagram. It keeps one representative per major physical segment and avoids variables that mostly duplicate the same measurement domain.

Candidate A columns, excluding `date` and the target `TE_8313B.AV_0#`:

`PT_8313C.AV_0#`, `PTCA_8322A.AV_0#`, `TE_8319A.AV_0#`, `TE_8303.AV_0#`, `ZZQBCHLL.AV_0#`, `AIR_8301A.AV_0#`, `YFJ3_AI.AV_0#`, `SXLTCYZ.AV_0#`

Candidate A is technically defensible because it preserves the core boiler structure with minimal redundancy: one upper-furnace pressure representative, one steam-side pressure representative, one economizer temperature, one air-preheater temperature, one steam-load flow variable, one oxygen variable, one fan-current variable, and one differential-pressure variable.

## Candidate B: Heatmap-Informed Physical Dataset
The family heatmaps show that some variables do not follow the simplest expected physical relationships. This does not invalidate the physical interpretation. In a controlled boiler, actuator action, operating regimes, time lags, load changes, and safety constraints can weaken or mask direct static correlations.

For this reason, Candidate B is selected as the main working dataset. It keeps all Candidate A variables and adds variables that preserve distinct heatmap-supported behavior not captured by the minimum physical set.

Variables added in Candidate B:

`TE_8332A.AV_0#`, `FT_8301.AV_0#`, `FT_8306A.AV_0#`, `YFJ3_ZD2.AV_0#`, `ZCLCCY.AV_0#`

The added variables have specific roles. `TE_8332A.AV_0#` preserves downstream outlet steam-temperature behavior. `FT_8301.AV_0#` and `FT_8306A.AV_0#` retain additional flow mechanisms beyond compensated steam flow. `YFJ3_ZD2.AV_0#` adds mechanical-condition information not fully represented by fan current. `ZCLCCY.AV_0#` adds a second differential-pressure view of gas-path resistance.

Candidate B is therefore the preferred base dataset for forecasting experiments. It is still reduced and physically interpretable, but it is less aggressive than Candidate A and better aligned with the observed multivariate structure.

## Candidate C: Control-Aware Forecasting Dataset
Candidate C is a second-level forecasting dataset. It keeps all Candidate B variables and adds the control-loop variables associated with desuperheating and outlet steam-temperature regulation.

Variables added in Candidate C:

`TV_8329ZC.AV_0#`, `YJJWSLL.AV_0#`

The motivation for Candidate C is that the boiler operates in a controlled environment. Control signals actively keep the process within safety and operating limits, which can mask simple sensor-to-sensor relationships. Including these variables may improve forecast performance because the model receives information about controller action, not only passive process measurements.

Candidate C must be interpreted carefully. If it outperforms Candidate B, the improvement may come from controller intelligence rather than from a better physical representation of the uncontrolled plant. For that reason, Candidate C is not the primary physical dataset; it is a control-aware forecasting comparison.

## Final Selection Strategy
- Candidate A documents the minimum feature set justified by first-principles physical segmentation.
- Candidate B is the main reduced dataset for forecasting because it combines physical segmentation with heatmap evidence.
- Candidate C is reserved for testing whether explicit control-loop information improves LSTM forecasting performance.

## Generated Dataset Files
- Candidate B is saved as `outputs/chinese_boiler_dataset/data/subset_B.csv`.
- Candidate C is saved as `outputs/chinese_boiler_dataset/data/subset_C.csv`.
- The family-level reduction rationale is saved as `outputs/chinese_boiler_dataset/data/family_reduction_table.csv`.