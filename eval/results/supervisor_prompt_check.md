# Supervisor prompt check (Oct 25, 2026)

Same three measurement windows sent to the deployed supervisor (Claude Haiku 4.5 on Bedrock). Real calls, not simulated.

## Before (first prompt)

Fatigue window (quality_mean 0.91, no flags, amplitude 0.58 / 0.61 of calibrated, 6 of 9 tilts near misses, trend -0.07 per minute): 3 of 3 runs chose `request_hold_still`, each after calling `get_direction_reliability`. The stated reasons contradicted the numbers ("Quality is good (0.91), so the issue is signal stability"). Asking a tired user to hold still does not make their movements larger: this is a wrong action, kept here as a failure case.

## Change

The system prompt now separates problem types (signal problem, shrinking movement, slow but accurate, unintended selections, nothing wrong) and the `request_hold_still` tool description says it is only for signal problems.

## After (current prompt), 2 runs per window

| Window | Action (both runs) |
| --- | --- |
| Fatigue (quality 0.91, amplitude 0.58 / 0.61, 6 of 9 near misses) | `adjust_sensitivity` scale 0.8 |
| Low light (quality 0.42, low_light 0.7, unstable 0.3) | `request_hold_still` |
| All good (quality 0.95, amplitude about 1.0, 5 selections) | `no_change` |

Latency 1.5 to 3.1 s per decision. The agent no longer queried extra evidence on these clear cases.
