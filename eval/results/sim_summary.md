# Simulated users study: SIMULATION, NOT PEOPLE

30 simulated users (seed 7), each calibrated by the real calibration analysis on a simulated clip, then a 10-message scripted task per arm, closed loop against the real Controller. Fatigue condition: reachable tilt shrinks linearly to 50% after 2 minutes. A target not reached within 45 s is skipped (counted as not achieved).

Parameter sources:

- tilt amplitude (roll): measured: developer sessions, roll p5/p95 about +/-22 deg during tilts
- tilt amplitude (pitch): measured: developer sessions, pitch p5/p95 about -11 / +13 deg
- pose jitter: measured: filtered pose second-difference median 0.15 (roll) to 0.35 (yaw) deg
- long blink length: measured: developer deliberate closures held 0.9 to 1.3 s (eye recordings)
- involuntary blinks: assumed: 15 per minute, 100 to 300 ms (typical adult values)
- reaction time, hold time: assumed
- dropouts, low-light spells: assumed: 0.5 dropouts and 0.2 low-light spells per minute
- limited-range and tremor profiles: assumed: up/down range 3 to 7 deg; jitter 1.5 to 2.5 deg

## normal

| Metric | A (Fixed menu, generic profile) | B (Dwell selection (no blink)) | C (Calibrated, no adaptation) | D (Calibrated + rule policy) | E (Calibrated + supervisor agent) |
|---|---|---|---|---|---|
| task_success_rate | 0.647 | 0.647 | 0.927 | 0.917 | 0.927 |
| false_triggers | 0.0 | 0.433 | 0.0 | 0.0 | 0.0 |
| selection_time_s_median | 14.12 | 12.273 | 13.234 | 13.39 | 13.267 |
| retries_per_correct | 0.269 | 0.171 | 0.474 | 0.502 | 0.482 |
| undos | 5.533 | 4.933 | 5.5 | 6.367 | 5.567 |
| highlight_timeouts | 0.133 | 0.0 | 0.3 | 0.8 | 0.3 |
| agent_decisions | 0.0 | 0.0 | 0.0 | 3.1 | 2.5 |
| agent_cost_usd | n/a | n/a | n/a | 0.0 | 0.009 |

Task success by user type:

| Type | A | B | C | D | E |
|---|---|---|---|---|---|
| limited_ud | 0.489 | 0.489 | 1.0 | 1.0 | 1.0 |
| tremor | 0.356 | 0.356 | 0.756 | 0.722 | 0.756 |
| typical | 0.983 | 0.983 | 1.0 | 1.0 | 1.0 |

## fatigue

| Metric | A (Fixed menu, generic profile) | B (Dwell selection (no blink)) | C (Calibrated, no adaptation) | D (Calibrated + rule policy) | E (Calibrated + supervisor agent) |
|---|---|---|---|---|---|
| task_success_rate | 0.23 | 0.23 | 0.59 | 0.75 | 0.777 |
| false_triggers | 0.0 | 0.0 | 0.0 | 0.0 | 0.033 |
| selection_time_s_median | 11.681 | 10.493 | 14.071 | 15.591 | 15.388 |
| retries_per_correct | 0.137 | 0.153 | 0.558 | 0.582 | 0.594 |
| undos | 7.933 | 7.567 | 9.133 | 6.933 | 7.733 |
| highlight_timeouts | 0.1 | 0.0 | 0.533 | 1.833 | 0.867 |
| agent_decisions | 0.0 | 0.0 | 0.0 | 3.433 | 2.8 |
| agent_cost_usd | n/a | n/a | n/a | 0.0 | 0.015 |

Task success by user type:

| Type | A | B | C | D | E |
|---|---|---|---|---|---|
| limited_ud | 0.156 | 0.167 | 0.678 | 0.9 | 0.978 |
| tremor | 0.022 | 0.022 | 0.222 | 0.367 | 0.311 |
| typical | 0.442 | 0.433 | 0.8 | 0.925 | 0.975 |

Calibration chose:

- typical: four_way x12
- limited_ud: binary_scan x9
- tremor: binary_scan x3, four_way x6

Limits: the user model is a simplification. It plans perfectly, does not learn, and its noise is synthetic. These numbers compare interface setups under the model; they are not evidence of how people with disabilities would perform.
