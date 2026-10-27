# HeadVoice: technical report

OpenCV AI Competition 2026. Research and education prototype, **not a medical device**.

Links: [website](https://d8wcckyu7sin3.cloudfront.net) · [judge replay page](https://cj0mdhx3ul.execute-api.us-east-1.amazonaws.com/judge) · [architecture](architecture.md) · [agent workflow and real trace](agent_loop.md) · [safety rules](../RULES.md) · [models](models.md)

## Summary

HeadVoice turns an ordinary webcam into a voice for people who cannot speak clearly or use their hands. The user tilts their head to highlight a message and holds a long blink to confirm it; the computer speaks it. OpenCV 5 does all the visual work on the user's computer: face detection (YuNet), 68 landmarks (LBF), head pose (solvePnP plus 2D geometry), and an eye-state classifier built with `cv2.HOGDescriptor` and `cv2.ml.SVM`. A 25-second calibration chooses the interface layout and personal thresholds from measured range, jitter and blinks. A supervisor agent on Amazon Bedrock adapts the interface during use from numeric OpenCV measurements; each of its decisions is validated on the device and its effect measured a minute later. Everything that decides what is spoken is plain, tested code on the device.

Evidence: 125 automated tests (including 11 safety rules); a deployed AWS stack with 9 passing end-to-end checks; 13 live sessions of the developer (22.5 minutes, 30 real agent decisions, published anonymized); a closed-loop simulated-users study (30 users, 5 interface setups, labeled as simulation); component evaluations of the eye-state model and the message agent; and a list of failures found and how each was handled.

## 1. Problem and users

People who cannot speak clearly or use their hands often depend on slow or costly input devices. Many can still move their head a little and blink, and a webcam is cheap and already in most homes. The core moment: saying "I need water" without help.

| Role | Needs |
| --- | --- |
| User | A fast, low-effort way to choose a message, and full control over what is spoken |
| Helper or caregiver | Simple setup, a way to fix things (re-center, calibrate, pick a layout), their own phrases for the person |
| Judge | Something they can run and understand quickly, with honest numbers |

## 2. What is new, and what is not

Not new: head and blink control, and ranked suggestions with confirm-before-speak, exist in other assistive tools.

New in this work:

- **Calibration chooses the interface.** Each direction's reliable range is measured against the user's own jitter; the layout (four-way or left/right scanning), thresholds, long-blink length and eye-open/closed levels follow from the data.
- **Vision-driven adaptation by a bounded agent.** OpenCV measurements (tilt size against the calibrated range, near misses, trend, quality) drive interface changes. The agent can query more evidence, sees its own past decisions with their measured outcomes, and every action is validated in code.
- **Honest measurement.** Failures found during development are kept and reported (section 8), including the agent's.

## 3. Architecture

Diagrams: [architecture.md](architecture.md).

- **Device (Python, `localhost:8765`).** Camera, OpenCV 5 pipeline, gesture state machine, browser UI with speech, offline phrase list, session monitor and the adaptation adapter, local logs. The UI has a landing page, a 4-step setup (welcome, camera check, optional calibration, practice), the message screen with a live camera window in the top-right corner, a helper panel, and a My messages page for the person's own phrases.
- **AWS (one CDK stack, us-east-1).** API Gateway HTTP API with a Lambda token authorizer and throttling; 9 Lambda functions from one container image holding OpenCV 5 headless contrib and the same `headvoice` package; Amazon Bedrock (Claude Haiku 4.5, Converse tool use) behind a Bedrock guardrail; S3 with a customer-managed KMS key; DynamoDB audit table; CloudWatch logs, embedded metrics and a dashboard; a CloudFront website with a Windows download.
- **Boundary.** The camera picture and every decision about what to say stay on the device. Only numeric summaries cross to AWS, plus an optional calibration clip after explicit consent, deleted as soon as it is read.

## 4. OpenCV 5 implementation

`opencv-contrib-python==5.0.0.93` (reports 5.0.0) on the device; `opencv-contrib-python-headless==5.0.0.93` in Lambda.

| Stage | OpenCV | Design and evidence |
| --- | --- | --- |
| Face detection | `cv2.FaceDetectorYN`, YuNet `2026may` | OpenCV 5's ONNX Runtime engine needs YuNet's dynamic-shape export; the 2023 model has a fixed input shape. Score threshold 0.6 because tilted faces score lower. |
| Landmarks | `cv2.face.createFacemarkLBF`, 68 points | opencv_zoo has no facial landmark model. YuNet misses faces rolled about 20 degrees, which is exactly a tilt gesture; on misses, LBF is refitted on a box carried from the previous landmarks (up to 15 frames, rejected if the fit drifts) before optical flow. On the first live session this removed all 162 flow frames that had dropped tracking 109 times in 125 s. |
| Head pose | `cv2.solvePnP` (6-point generic face, camera matrix from image size) for pitch and roll; yaw from the nose offset along the eye line | 6-point PnP yaw flipped between two near-equal solutions (155 one-frame jumps over 8 degrees with roll and pitch unchanged). Geometric yaw is single-valued and unaffected by roll. On an 11-minute session, false "turned away" frames fell from 14.3% to 2.5%. |
| Smoothing and neutral | One Euro filter; neutral learned from the first steady second, then follows slow posture drift (20 s time constant) only near rest | Stability is judged on the filtered pose (median jitter 0.35 degrees). |
| Eye state | `cv2.HOGDescriptor` + linear `cv2.ml.SVM` on 48x24 eye crops aligned to the eye corners | LBF eyelid landmarks barely move when eyes close: eye aspect ratio separated open from closed at chance level (AUC 0.34 to 0.75). Trained on 6,054 labeled crops from three guided recordings; detected 24 of 24 held-out long blinks; 0.17 ms per frame. |
| Quality and recovery | brightness, contrast, face size, stability, turned away; `cv2.createCLAHE` on the face region with hysteresis; `cv2.calcOpticalFlowPyrLK` fallback (10 frames, never confirms); frozen-feed detection | A phone-webcam driver repeated one frozen frame; 10 identical frames now set quality to 0. |
| Cloud | `cv2.VideoCapture` on the consented calibration clip, same pipeline and analysis | Device and cloud give the same result by construction. |

Speed on a laptop CPU at 640x480 with an HP W100 webcam: camera-limited 30 fps; detection 10 to 11 ms, landmarks 4.5 ms, pose 0.6 ms, eye state 0.17 ms per frame.

## 5. Interaction and safety

The gesture state machine (`interaction/state_machine.py`) is a pure function of the previous state, the frame's measurements and the profile. States: idle, tracking, highlighting, armed, speaking, recovering. A tilt must be held past the threshold (300 ms by default) and return toward neutral before it counts again. A message is spoken only after both eyes stay closed for the long-blink time (700 ms by default) on a single highlighted message, on usable frames, starting after the eyes were seen open. A frame is usable only with a face, model landmarks (not optical flow), a head facing the screen and quality at least 0.5. Two quick blinks undo; a long blink with nothing highlighted shows other messages; eye closure is ignored while the head is pitched down.

Eleven safety rules ([RULES.md](../RULES.md)) are enforced in code and covered by tests: code owns the state, fail closed, debounce, confirm before speaking, undo, stale-result guard, bounded agent, tool tiers, action cap, offline fallback, privacy.

## 6. Agent design

Details and diagrams: [agent_loop.md](agent_loop.md).

**Supervisor.** Triggered by the session monitor on a quality drop (any time) or, after the session's first minute, on an amplitude decline (at least 3 near misses and 6 deliberate attempts), an undo spike, 5 selections or 60 s. A tilt attempt counts only if it passes half the threshold and is held 0.2 s. The request holds numbers only: the window summary, current parameters, whether a helper locked the layout, and the last 3 decisions with their measured outcomes. The agent may first query a longer window or per-direction reliability, answered from an evidence pack the device sends, then makes one change: `adjust_sensitivity` (added during development, because the planned tools could not lower tilt thresholds, the main fatigue adaptation), `adjust_timing`, `set_layout`, `request_hold_still`, `request_recalibration` (the user must accept) or `no_change`. At most 3 tool calls per invocation. The device validates tier, bounds, epoch, action cap and layout lock, applies the change with a one-step rollback, and records `outcome_after` 60 s later; the next decision waits for that outcome.

**Message agent.** Ranked phrasebook IDs with an abstain path, forced through a tool schema and validated with pydantic; free text and unknown IDs are rejected and the device falls back to the offline list. It cannot speak.

**Prompt failure found and fixed with real calls.** With the first prompt, a clear fatigue window (quality 0.91, amplitude 0.58 and 0.61, 6 of 9 tilts short) produced `request_hold_still` in 3 of 3 runs, with reasons contradicting the numbers. After the prompt separated signal problems from shrinking movement, the same window gave `adjust_sensitivity` 0.8, a low-light window `request_hold_still`, and a healthy window `no_change`, in 2 of 2 runs each ([supervisor_prompt_check.md](../eval/results/supervisor_prompt_check.md)).

**Real trace.** In the developer's 7.9-minute live session the agent lowered sensitivity at 60 s, citing measured tilt sizes of 0.54 and 0.46 of range and 21 of 34 attempts falling short. At 120 s, with down still failing after that change, it switched the layout; when quality dropped at 170 s it asked the person to hold still. Outcomes were measured 60 s after each change. In another live session the device rejected an out-of-bounds request (sensitivity 0.4). Full table: [agent_loop.md](agent_loop.md#real-trace-the-developers-own-session).

## 7. AWS deployment

Infrastructure as code: `infra/headvoice_stack.py` (CDK v2). Deploy and teardown steps are in the [README](../README.md#aws-deployment). `scripts/smoke_cloud.py` checks 9 endpoints and guards against the live stack (including that the authorizer rejects a bad token with 403 and calibration refuses keys outside its prefix): 9 of 9 pass.

Security: bearer token in Secrets Manager checked by a Lambda authorizer; least-privilege IAM per function (only the agent and replay functions may call Bedrock; only calibration may touch `calibration-clips/`); S3 blocks public access, enforces TLS and uses a customer-managed KMS key with rotation; DynamoDB encrypted; Bedrock guardrail (content filters, prompt-attack filter, PII anonymization, denied medical-advice topic); stage throttling (10 requests per second, public judge replay 0.5); outbound payload check on the device and again in the Lambda.

Measured on the live stack: container image 1.46 GB; Lambda cold start 430 to 620 ms; supervisor decisions 3.3 to 3.6 s median end to end in live sessions (almost all of it Bedrock); about 4,800 input and 190 output tokens per live decision; 0.17 USD of Bedrock for all 30 live decisions (0.006 USD per decision at the listed Haiku 4.5 price of 1 and 5 USD per million tokens; confirm current prices on the AWS pricing page). Agent replays of long sessions exceed API Gateway's 30 s limit, so the judge page serves precomputed agent replays cached in S3; failed calls are never cached.

## 8. Evaluation

### 8.1 Live sessions of the developer (real, n=1)

13 sessions, 22.5 minutes, one healthy adult (the developer, not a target user), webcam in an ordinary room, agent on. Face found in 98.8% of frames; 40 messages spoken; 13 undos. The sessions are published anonymized (no landmarks or face boxes) on the judge page as `developer_*` and can be replayed through any setup.

| | Agent decisions | Checked evidence first | Outcome measured | Rejected by the device | Actions |
| --- | --- | --- | --- | --- | --- |
| Earlier adaptation rules (9 sessions) | 11 | 11 | 4 | 0 | 7 layout, 2 sensitivity, 1 hold still, 1 timing |
| Current rules (3 sessions) | 19 | 17 | 11 | 1 (sensitivity 0.4, out of bounds) | 10 hold still, 5 sensitivity, 4 layout |

Under the earlier rules the agent switched layouts 10 to 30 s into sessions, on uncalibrated ranges and glances counted as tilts (failure 9). Under the current rules, decisions driven by tilt measurements came only after the first minute, and most decisions were "hold still" when quality dropped. The one attempt to change a setting before 60 s came on a quality-drop trigger in the 1.5-minute session (sensitivity 0.4 at 41 s) and was rejected by the device as out of bounds, so nothing changed. The sessions were free use rather than scripted tasks (3 short scripted tasks: 2 targets reached, 1 wrong selection), so they support the trace and behavioural findings, not task-success statistics.

Replaying all 13 sessions through setups A to D (`eval/run_eval.py`, open loop, [summary.md](../eval/results/summary.md)): in the 2-second windows where signal quality was low (8.7 per session on average), the blink-confirmed setups (A, C, D) selected nothing in 100% of them; dwell selection (B) selected something in 6% of them. This is the fail-closed rule holding on real data, and the main reason blink confirmation was kept over dwell.

### 8.2 Simulated users study (simulation, not people)

`eval/sim_users.py`: 30 simulated users (12 typical, 9 with limited up and down range, 9 with tremor-like noise) with parameters anchored to the developer's measured tilt sizes, jitter and blink lengths; assumed values are listed in [sim_summary.md](../eval/results/sim_summary.md). Each user is calibrated by the real calibration analysis on a simulated clip, then does a 10-message scripted task in every setup, closed loop against the real Controller: it reads the screen and reacts to layout changes. Setup E calls the deployed supervisor on Bedrock. Fatigue condition: reachable tilt shrinks to 50% over 2 minutes. A target not reached within 45 s is skipped.

Setups: A fixed menu with generic profile, B dwell selection (no blink), C calibrated, D calibrated + rule policy, E calibrated + supervisor agent.

| Task success | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| Normal | 0.65 | 0.65 | 0.93 | 0.92 | 0.93 |
| Fatigue | 0.23 | 0.23 | 0.59 | 0.75 | 0.78 |

- **Calibration matters most.** It chose left/right scanning for all 9 limited-range users and four-way for all 12 typical users; limited-range users went from 0.49 with the fixed menu to 1.0.
- **Adaptation recovers much of the fatigue loss** (0.59 to 0.75 with the rule policy, 0.78 with the agent). The agent did better than the rule policy for typical (0.975 against 0.925) and limited-range users (0.98 against 0.90) and worse for tremor-like users (0.31 against 0.37).
- **The stricter adaptation rules have a cost.** An earlier version adapted sooner: tremor-like users under fatigue reached 0.54 to 0.61, but the agent requested out-of-bounds sensitivity 22 times and switched layouts 14 times. With the current rules: 0 out-of-bounds requests and 2 layout switches in the fatigue condition. The live sessions showed why the stricter rules were needed.
- **False selections:** blink confirmation produced one false selection in 30 fatigue sessions with the agent (0.03 per session) and none in other blink setups; dwell selection (B) produced 0.43 per session in the normal condition.
- Agent cost 0.009 to 0.015 USD per simulated session; median decision latency 3.5 to 4.0 s.

Limits: the user model plans perfectly, does not learn, and its noise is synthetic; involuntary blinks trigger undo more often than in the live sessions (about 5 undos per task). The study compares setups under the model, not how people would perform.

### 8.3 Component evaluations (real data)

**Eye-state model** (labeled eye crops of the developer, train on two recordings, test on the third):

| Held-out recording | Long blinks detected | Eyes-open prompts with a false long blink |
| --- | --- | --- |
| Glasses 1 | 8/8 | 2/9 (looking down) |
| Glasses 2 | 8/8 | 0/9 |
| No glasses | 8/8 | 2/9 (squint, looking down) |

**Message agent** (real Bedrock calls; 40 scenarios written by the team, 26 clear and 14 ambiguous):

| Variant | Top-1 (clear) | Top-3 (clear) | Abstained (ambiguous) | Abstained (clear) | Invalid output |
| --- | --- | --- | --- | --- | --- |
| Fixed order | 0.08 | 0.19 | 0.00 | 0.00 | 0.00 |
| Careless prompt | 0.85 | 1.00 | 0.43 | 0.00 | 0.00 |
| Evidence-first prompt | 0.81 | 0.92 | 0.93 | 0.04 | 0.00 |

Evidence-first gives up a little top-1 accuracy for far better abstention when the evidence does not support a message. Run 1, before a schema fix, had 10% and 20% invalid outputs, all from an unstated length limit ([messages_summary_run1.md](../eval/results/messages_summary_run1.md)).

**Tracking fixes on logged sessions:** replaying the first live session (125 s) through the fixed state machine cut drops to idle from 109 to 9 and false undos from 9 to 2.

Not measured: NASA-TLX workload, calibration stability across two clips per person, and a live comparison of setups C and E with several people.

## 9. Failure cases

All found during development on real runs, and how each was handled:

1. A phone-webcam driver repeating one frozen frame made the app unresponsive. Detected now (`frozen_feed`) and fails closed; the camera picker marks frozen cameras.
2. The face detector loses faces tilted about 20 degrees. Fixed with landmark-box tracking.
3. Generic-model PnP yaw is bistable. Replaced by 2D geometry for yaw.
4. LBF eyelid landmarks cannot tell open from closed eyes. Replaced by the eye-state model.
5. Eye state fails when looking down and when squinting. Head-down pitch is handled; eye-only downward gaze is not.
6. The supervisor prompt misread fatigue as a signal problem. Fixed (section 6).
7. Message-agent outputs were rejected for an unstated schema limit. Fixed.
8. In simulation the agent repeatedly requested out-of-bounds sensitivity (22 rejections), and one applied decision stated a wrong comparison (0.77 called "below 0.75"). Device-side bounds and outcome checks kept these harmless; the current rules removed the out-of-bounds requests.
9. Live, the agent switched the layout 33 s into a session, on uncalibrated ranges and glances counted as tilt attempts. Fixed with a one-minute warm-up, deliberate-attempt detection, sensitivity before layout, and a helper layout lock.
10. In the developer's 7.9-minute session the layout flip-flopped three times, and four setting changes came on quality-drop triggers, where only "hold still" or "no change" should be allowed. **Not yet fixed**; the next step is a device-side check that allows only those two actions on a quality-drop trigger, and a minimum time between layout changes.
11. The agent requested sensitivity 0.4 in a live session, 41 s in, on a quality-drop trigger; the device rejected it as out of bounds. This is also an instance of failure 10.
12. Agent replays of long sessions exceed API Gateway's 30 s limit. The judge page serves precomputed replays; a temporary Bedrock outage (`ServiceUnavailableException` on the US route) showed that failed replays must not be cached, which is now enforced.

## 10. Limitations

- Research and education prototype, not a medical device; not tested with people who rely on such a device.
- Replay of logged sessions is open loop and cannot show how a person would react to a changed layout.
- The eye-state model was trained on one person's eyes; calibration re-thresholds it per user but does not retrain it.
- The adaptation agent can still flip layouts and act on quality-drop triggers (failure 10).
- Sensitive to strong backlight; the generic 3D face model limits absolute pose accuracy, so decisions use pose relative to the user's neutral.
- Message scenarios were written by the team.
- The LBF landmark model has no license file in its source repository.

## 11. Responsible use and privacy

The camera picture never leaves the computer and is never stored; the camera window is served on localhost only. Cloud requests carry only numbers and short labels, checked on the device and again in the Lambda. The optional calibration clip needs explicit consent, is encrypted, read once and deleted. Local logs hold measurements and chosen messages and never upload automatically; shared samples have landmarks and face boxes removed (`scripts/make_sample.py`). My messages stay on the computer. The eye-state model file keeps only the SVM weights; OpenCV's stored training vectors were removed. The agent cannot speak, send alerts or see video, enforced in code and tested. Nothing is spoken without the user's own confirmed gesture.

## 12. Reproducing results

```sh
uv sync --all-groups && uv run python scripts/get_models.py
uv run pytest                                          # 125 tests, including the 11 safety rules
uv run python scripts/build_site.py
npx aws-cdk@2 deploy HeadVoice --outputs-file cdk-outputs.json
uv run python scripts/smoke_cloud.py                   # 9 end-to-end checks
uv run python eval/sim_users.py --agent api            # simulated users study
uv run python eval/run_message_eval.py --agent api     # message agent
uv run python scripts/warm_replay_cache.py             # precompute agent replays for the judge page
```
