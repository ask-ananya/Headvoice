# Progress

## Oct 27, 2026: developer sessions published, adaptation tightened

- Adaptation fixes after a live session switched layout at 33 s: one-minute warm-up, deliberate tilt attempts only (half the threshold, held 0.2 s), at least 6 attempts before an amplitude trigger, sensitivity before layout in the prompt, and a helper layout picker that locks the layout.
- 12 developer sessions (format 2, agent on) anonymized and published on the judge replay page next to 12 labeled simulated sessions. Real agent trace documented in docs/agent_loop.md.
- Agent replays cached in S3 (API Gateway 30 s limit); `scripts/warm_replay_cache.py` precomputes them.
- Simulation rerun with the new rules (report section 7).

## Oct 27, 2026: replay page populated

The judge replay page now lists 12 sessions from the simulated-users study (2 users each of typical, limited up/down range and tremor-like noise, normal and fatigue), exported with `eval/sim_users.py --export-samples data/sample` and uploaded to `sample/`. Each file is marked simulated in its header and labeled "Simulated user" on the page. The developer will not record n=1 sessions, so there is no recording of a person in the evaluation; the report says so.

## Oct 27, 2026: customer-facing app, website, simulated users study

- New UI (reference style: warm off-white, rose accent, Source Sans 3, light and dark): landing page, 4-step setup (welcome, camera check with picker and live preview, optional calibration, practice), message screen with a live camera window top right, blink ring around the highlighted card, guidance strip, helper panel with a Research section. My messages tab for the user's own phrases (stored locally, shown first).
- Website on CloudFront (landing in "Get HeadVoice" mode, download page with a Windows zip and one-click launcher, privacy page). Clean-folder install from the zip tested.
- Eye model file stripped of OpenCV's stored training support vectors (features of the developer's eye crops): identical predictions, 7.6 MB to 6.3 KB.
- No volunteers: protocol rewritten for n=1 (the developer), and a simulated-users study (30 users, arms A to E, real calibration and real supervisor calls). Results in `eval/results/sim_summary.md` and the report.

## Oct 25, 2026: Phases 3 to 6 built, AWS deployed

- Phase 3: calibration protocol and analysis (shared device and Lambda), in-app calibration with consent screen and tone cues, replay engine driving the real Controller (replay reproduces live events exactly, tested), fatigue simulation, dwell confirm mode for baseline B.
- Phase 4: CDK stack deployed to us-east-1 (S3 with KMS and clip lifecycle, DynamoDB audit, Secrets Manager token, Lambda token authorizer, HTTP API with throttling, Bedrock guardrail, 9 container Lambdas from one image, CloudWatch dashboard). `scripts/smoke_cloud.py`: 9 of 9 checks pass. Deploy issues found and fixed: numpy wheels too new for the Lambda base (pinned 2.2.6 everywhere), API stage route settings created before routes (explicit dependency).
- Phase 5: session monitor (tilt attempts, near misses, amplitude vs calibrated range, trend), rule baseline, supervisor (Converse tool use, evidence pack, max 3 calls), message agent (forced tool schema, strict validation), adapter loop with outcome window and traces to JSONL and /audit. Added tool `adjust_sensitivity` (bounded 0.5 to 1.0).
- Supervisor prompt failure found and fixed with real calls (eval/results/supervisor_prompt_check.md).
- Adapter fix: after an applied change, decisions wait for its outcome window (the live check showed it re-deciding on windows still dominated by pre-change tilts).
- Phase 6: message agent evaluated through Bedrock (40 scripted scenarios). Run 1 exposed a schema mismatch (20% invalid); run 2 after the fix has 0% invalid. Arm evaluation tooling ready; no eligible sessions yet (all earlier logs use the old pipeline format).

Still needed from the team: new recordings with the current pipeline (normal, adverse, small movements, scripted tasks), a live agent session for the real trace, the video, and consent to publish anonymized samples for the judge page.

## Oct 25, 2026: second session analysis, yaw and blink rework

Session `session_20261025_020229.jsonl` (11 min): 18 messages spoken, no flow frames (box tracking worked), median usable run 14 frames (was 2). Remaining problems and fixes:

1. **PnP yaw flips between two solutions** (155 one-frame jumps over 8 degrees with roll and pitch unchanged), which caused most `turned_away` and low-quality frames. Yaw now comes from 2D geometry (nose offset along the eye line over eye-corner distance); pitch and roll stay on solvePnP. Replayed on the session's landmarks: turned_away frames 14.3% to 2.5%, yaw-limited quality below 0.5 from 16.2% to 3.8%.
2. **Posture drift** (roll about -5 degrees and yaw offsets building up): the neutral now follows slow drift (20 s time constant) only while the head is near rest, never during a tilt.
3. **Blinks**: EAR from LBF was at chance level on labeled data. Replaced by an eye-state model (HOG + `cv2.ml` SVM on eye crops), 24 of 24 long blinks detected on held-out recordings. See `docs/models.md`. Eye closure is ignored while the head is pitched down.

## Oct 25, 2026: stability fixes from the first real session log

Session `session_20261025_014933.jsonl` (125 s, HP W100, 29.8 fps, no frame gaps over 63 ms, face 230 px wide, low light on 1% of frames): the camera and processing speed were fine. The instability came from:

1. YuNet misses faces rolled about 20 degrees, which is a tilt gesture. All 162 flow frames were mid-tilt, and each one dropped tracking to idle (109 drops in 125 s). Fix: detector threshold 0.6, and when YuNet misses, refit LBF on a box carried from the previous landmarks (max 15 frames, rejected if the fit moves away), before falling back to flow (`tracked_box` flag).
2. Tilts read as large yaw (corr(yaw, roll) 0.61; the 2D nose offset also follows roll, so part of it is real head rotation that the generic model exaggerates). 25% of tilts were flagged turned_away. Fix: turned away = yaw beyond 35 degrees from neutral when the head is level, 50 when tilted 8 degrees or more.
3. LBF eyelid points move little: deliberate closures only reached openness 0.62 to 0.70, with frame-to-frame flicker around the 0.7 threshold. Every long blink broke into 31 to 141 ms pieces (0 of 6 confirmed), and flicker produced 9 false double-blink undos. Fix: 3-frame median on openness, per-eye hysteresis (closes below 0.75, reopens above 0.85), closures under 60 ms ignored for undo.
4. Any single bad frame reset progress. Fix: 300 ms grace before tracking drops to idle; a highlight survives a dropout shorter than the recovery timeout. Safety kept: any bad frame cancels an armed blink, and after a dropout or speech the eyes must be seen open before a new blink counts (new tests in test_safety.py).

Replay of the same session through the new state machine (box tracking cannot be replayed without video): drops to idle 109 to 9, false undos 9 to 2 (both with nothing highlighted), long blinks confirmed 0 to 2 (the deliberate attempts at 244 s and 259 s). One attempt (267 s) still ended at 328 ms. Blink detection from LBF remains the weakest signal; calibration should set the thresholds per user, and an eye-region appearance measure is the next step if it stays unreliable.

## Oct 25, 2026: first live camera run, fixes

Symptom: no response to tilts or blinks. Causes found and fixed:

1. **Frozen camera.** Camera 0 on the dev laptop is the iVCam phone-webcam driver. With the phone not streaming it repeats one image (frame difference exactly 0). The pipeline now flags `frozen_feed` after 10 identical frames (quality 0, no neutral learned), and the app and live debug pick the first camera with a live picture. `scripts/list_cameras.py` shows each camera's status. The real webcam (HP W100) was at index 1.
2. **Absolute yaw bias.** The generic 6-point face model read near-frontal faces as 40 to 46 degrees yaw (all PnP solvers agreed, reprojection error under 3 px, so it is model mismatch, not a solver bug). Absolute yaw tripped `turned_away` on every frame. Now pose is relative to the user's neutral everywhere, including the turned-away check. If no neutral exists, the pipeline learns it from the median of the first steady second (30 frames) and fails closed until then. The operator can reset it with `N`.
3. **Stability too strict.** Raw PnP yaw noise on the W100 is large (median second difference 3.6 degrees; pitch 1.9, roll 1.3). Stability is now measured on the One Euro filtered pose (median 0.35 degrees) with thresholds rescaled from that measurement.

After the fixes, live on the W100: 30 fps (camera limit), detect 11 ms, landmarks 4.5 ms, neutral learned in 0.95 s, median quality 1.0 when facing the screen, turning away flagged. Eye openness dipped only to 0.62 to 0.70 in 10 s of unscripted use: deliberate long blinks still need checking against the 0.7 threshold.

The app's page shows no camera video to the user by design. The inspector now has a local preview (`/preview.mjpg`, localhost only).

## Oct 25, 2026: Phases 0 to 2 (code complete, live check pending)

### Phase 0: setup and verification

- Python 3.11, `uv` project, `uv.lock` and `requirements.lock`, ruff and pre-commit config, pytest.
- Version control: none. The repo owner asked not to initialise git for now; the plan's "commit after each task" rule is paused.
- OpenCV: `opencv-contrib-python==5.0.0.93` reports version 5.0.0. `FaceDetectorYN`, `cv2.face`, `solvePnP`, `calcOpticalFlowPyrLK` and CLAHE are all present.
- Landmark path chosen: FacemarkLBF 68 points (plan path 2). Path 1 has no model in opencv_zoo. Details in `docs/models.md`.
- Finding: OpenCV 5 needs the `face_detection_yunet_2026may.onnx` model (dynamic input shape). The 2023mar model has a fixed input shape, and the OpenCV 5 ONNX Runtime engine does not accept that.
- `scripts/verify_opencv.py` passes on the OpenCV sample image. Webcam frame captured, but no face was in view at the time, so the webcam face check is still open.
- Bedrock (us-east-1): Converse with tool use works on `us.anthropic.claude-haiku-4-5-20251001-v1:0` (1.7 s) and `us.anthropic.claude-sonnet-4-6` (2.1 s). Claude 5.x models return "not available for this account". Default set to Haiku 4.5.
- AWS credentials are the account root user. Replace with a least-privilege IAM user before deployment and before sharing anything with judges.

### Phase 1: vision pipeline

- Detector, landmarks, pose (solvePnP + One Euro filter), blink (EAR normalized by a rolling per-eye baseline), quality score and flags, CLAHE trigger with hysteresis, Lucas-Kanade fallback (max 10 frames, flow frames can never confirm).
- Measured on the sample image (laptop CPU): detect 7.6 ms, landmarks 3.6 ms, pose 0.5 ms per frame.
- `scripts/live_debug.py` and the JSONL logger.
- Tests: pose recovers known rotations within 0.5 degrees, sign conventions, One Euro behaviour, blink events from synthetic series, one-eye rejection, long-blink threshold, quality flags, CLAHE hysteresis, bounded flow.

### Phase 2: interaction and local app

- Pure state machine with idle, tracking, highlighting, armed, speaking, recovering. Layouts four_way, two_choice, binary_scan. "None of these" is a long blink with nothing highlighted. Undo is a double blink by default, or tilt down in left/right layouts.
- FastAPI app, WebSocket UI with highlight, armed progress ring, speech, status bar and inspector.
- `RULES.md`; `tests/test_safety.py` covers all 11 invariants.
- 76 tests pass, ruff clean.

### Design decisions to review

- Left and right use head roll (ear toward shoulder) by default, so a turned head stays a fail-closed condition. `lr_axis: "yaw"` in the profile switches to turning. Sign flips (`lr_sign`, `ud_sign`) handle mirrored cameras.
- "None of these" needs no screen slot: a long blink with nothing highlighted. It never speaks.
- Tilt thresholds (12 degrees left/right, 10 degrees up/down), the 0.7 closed-eye ratio and the 700 ms long blink are defaults until calibration (Phase 3) sets them from data.

### Open: go/no-go (plan section 11), deferred by the owner until after Phase 2

Run `scripts/live_debug.py` and the app with Akshat and one other person in ordinary room light. Check:

1. Left and right tilts are counted reliably, and up and down if possible.
2. A deliberate long blink is detected and normal blinks are not. Main risk: LBF can under-fit closed eyes, so watch the eye L/R readout. Closed eyes should fall below 0.7.
3. Covering the camera or turning away never speaks.

### Human tasks now

- Go/no-go test above.
- Start volunteer recordings (`eval/protocol.md`) once the go/no-go passes.
- Replace root AWS credentials.
