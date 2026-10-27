# HeadVoice: Implementation Plan

Handoff document for Claude Code. Read all of it before writing code.

## 0. Read this first

### What we are building

HeadVoice is a webcam communication aid. A person who cannot speak clearly or use their hands picks a message with head tilts, confirms it with a long blink, and the app speaks it aloud. Per-user calibration chooses the interface layout, and an agent adapts the interface from what OpenCV 5 measures. Nothing is ever spoken without the user's own confirmed gesture.

Entry for the OpenCV AI Competition 2026 (Devpost, AWS-sponsored).

### Hard deadline

- **Oct 26, 2026, 11:45 pm PDT = Oct 27, 2026, 12:15 pm IST.** The page also says 11:59 pm in one place. Plan for the earlier time.
- Target: everything submitted by **Oct 27, 9:00 am IST**, leaving a 3 hour buffer.

### Competition hard requirements (non-negotiable)

1. OpenCV 5 performs substantive image or video analysis.
2. A meaningful component runs on AWS.
3. Submission includes: technical report, judge-accessible repo, pinned dependencies with build/deploy/test steps, architecture diagram, working web endpoint or arranged live demo, video of 5 minutes or less, evaluation evidence with failure cases.
4. For the Agentic Vision Award: an agent workflow diagram, and a trace showing OpenCV output changed a later decision, tool call or action. A chatbot explaining a fixed vision result does not qualify.

### Working rules for Claude Code

- **Verify, do not assume.** OpenCV 5 packaging and APIs, Bedrock model IDs, and Lambda container limits must be checked against the installed version or official docs before relying on them. If something in this plan turns out to be wrong, say so and propose the fix.
- **Tasks marked `[HUMAN]`** need Akshat (AWS credentials, Bedrock model access, recording volunteers, filming the video). Stop and ask when you reach one. Do not fake it.
- **Never fabricate results.** Evaluation numbers come only from real runs. If a run did not happen, the report says so.
- **Safety invariants (section 6) are tested code, not comments.** Every one has a unit test.
- **Commit after each task** with a clear message. Keep `main` runnable.
- **No em dashes** in any user-facing text (README, report, UI strings, video script). Use colons, commas or full stops.
- **Keep it small.** When a choice exists, pick the simpler option that meets the requirement. Respect the cut order in section 12.

---

## 1. Architecture overview

```
DEVICE (Python, local)                                AWS (CDK stack)
+------------------------------------------+          +-------------------------------------+
| Webcam                                   |          | API Gateway (HTTP API, token auth)  |
|  -> OpenCV 5 vision pipeline             |          |  /calibrate  -> Lambda (container,  |
|     face, landmarks, pose, blink,        | numbers  |                 OpenCV 5)            |
|     quality, optical-flow fallback       |  only    |  /replay     -> Lambda (container,  |
|  -> Gesture state machine (code decides) | -------> |                 OpenCV 5)            |
|  -> Session monitor (signal summaries)   |          |  /supervise  -> Lambda -> Bedrock    |
|  -> Local web UI (FastAPI + WebSocket)   | <------- |  /suggest    -> Lambda -> Bedrock    |
|  -> Speech (browser speechSynthesis)     | profiles,|  /audit      -> DynamoDB             |
|  -> Offline phrase list                  | tool     | S3 (KMS): calibration clips (deleted |
|  -> Landmark logger (JSONL)              | calls    |   after processing), replay datasets |
+------------------------------------------+          | DynamoDB: audit trail                |
                                                      | CloudWatch: logs, metrics, dashboard |
                                                      +-------------------------------------+
```

Raw video never leaves the device in normal use. The single exception is the calibration clip, uploaded only with explicit consent and deleted after processing.

The app must work fully offline with the default profile and the fixed phrase list.

---

## 2. Tech stack

| Area | Choice | Notes |
| --- | --- | --- |
| Language | Python 3.11 | Pin exact version in `.python-version` |
| Vision | OpenCV 5 (`cv2`) | Verify package name and version in Phase 0 |
| Local server and UI | FastAPI, uvicorn, WebSocket, plain HTML/JS | No frontend framework |
| Speech | Browser `speechSynthesis` | Works offline; Amazon Polly is optional stretch |
| Validation | pydantic v2 | All agent I/O and config |
| Infra | AWS CDK v2 (Python) | One stack, `infra/` |
| Cloud compute | Lambda container images | Fargate only if Lambda limits block OpenCV |
| LLM | Amazon Bedrock Converse API with tool use | Model ID in config, verify access first |
| Storage | S3 (SSE-KMS), DynamoDB | Private, encrypted |
| Tests | pytest | Unit tests for every safety invariant |
| Dependency pinning | `uv` or `pip-tools` lock file | Lock file committed |

---

## 3. Repository layout

```
headvoice/
  README.md                 # build, run, deploy, test, demo steps
  RULES.md                  # one-page safety rules (section 6)
  plan.md                   # this file
  pyproject.toml
  requirements.lock
  config/
    default_profile.json    # ships with app, calibration optional
    phrasebook_en.json
    agent_bounds.json       # allowed ranges for agent-tunable params
  src/headvoice/
    vision/
      detector.py           # face detection
      landmarks.py          # landmark model wrapper
      pose.py               # solvePnP head pose + filtering
      blink.py              # eye openness, blink events, duration model
      quality.py            # signal quality score, CLAHE trigger
      flow.py               # Lucas-Kanade fallback tracking
      pipeline.py           # frame -> FrameFeatures
    interaction/
      state_machine.py      # gesture states (section 5)
      gestures.py           # tilt and blink event extraction, debounce
      layouts.py            # four_way, binary_scan, two_choice
      selection.py          # menu model, undo, none-of-these
    agent/
      monitor.py            # rolling signal summaries per window
      supervisor_client.py  # calls /supervise, applies validated tool calls
      suggest_client.py     # calls /suggest
      rule_policy.py        # deterministic baseline adaptation policy
      tools.py              # tool schemas, tiers, bound checks
      trace.py              # decision trace records
    calibration/
      record.py             # consent prompt, clip capture
      analyze.py            # shared by local and Lambda
    io/
      logger.py             # JSONL landmark/event logs
      schemas.py            # pydantic models for logs, profiles, agent I/O
      offline.py            # offline phrase fallback
    app/
      server.py             # FastAPI app, WebSocket
      static/index.html
      static/app.js
      static/style.css
    replay/
      engine.py             # logs -> features -> state machine -> events
      fatigue_sim.py        # amplitude decay and noise augmentation
  cloud/
    lambdas/
      calibrate/            # Dockerfile + handler, imports headvoice.calibration
      replay/               # Dockerfile + handler, imports headvoice.replay
      supervise/            # Bedrock supervisor agent
      suggest/              # Bedrock message agent
      audit/                # read audit trail for inspector
  infra/
    app.py                  # CDK app
    headvoice_stack.py
  eval/
    protocol.md             # volunteer recording protocol
    run_eval.py             # all arms, writes results/
    metrics.py
    scenarios/              # message-agent test scenarios with ground truth
    results/                # generated, committed for judges
  data/
    sessions/               # recorded landmark logs (gitignored except sample/)
    sample/                 # anonymized replay dataset for judges
  docs/
    architecture.md         # mermaid diagrams
    agent_loop.md           # agent workflow diagram + trace example
    report.md               # technical report
    video_script.md
  tests/
```

---

## 4. Vision pipeline (on device)

Output of the pipeline for every frame is a `FrameFeatures` record:

```python
class FrameFeatures(BaseModel):
    t: float                    # seconds since session start (monotonic)
    frame_idx: int
    face_found: bool
    bbox: tuple[int,int,int,int] | None
    landmarks: list[tuple[float,float]] | None
    landmark_source: Literal["model", "flow", "none"]
    yaw: float | None           # degrees, filtered
    pitch: float | None
    roll: float | None
    eye_open_l: float | None    # normalized eye openness (e.g. EAR)
    eye_open_r: float | None
    quality: float              # 0..1
    quality_flags: list[str]    # e.g. ["low_light", "turned_away", "unstable"]
    clahe_applied: bool
```

### 4.1 Phase 0 verification (must happen first)

Write `scripts/verify_opencv.py` that prints and checks:

- Installed OpenCV version is 5.x. If no OpenCV 5 wheel installs cleanly, report it and stop for a decision.
- Which face detector works: try `cv2.FaceDetectorYN` (YuNet ONNX) first.
- Which landmark path works for **eye-lid landmarks** (needed for blink) and enough points for pose. Candidates, in order:
  1. A facial landmark model loaded through `cv2.dnn` (ONNX), if one runs correctly.
  2. `cv2.face.createFacemarkLBF` with the LBF model, if the contrib module exists in the OpenCV 5 build.
  3. Fallback: YuNet's 5 points for pose, plus eye-region analysis inside OpenCV for blink (eye ROI intensity, edges, or optical flow on the lid). Lower quality, document the limit.
- Record model file names, sources and licenses in `docs/models.md`. Download script in `scripts/get_models.sh`.

Go/no-go is in section 11.

### 4.2 Head pose

- `cv2.solvePnP` with a generic 3D face model (6 to 14 reference points, depending on landmark set).
- Camera matrix from image size (focal length approximated as image width, principal point at center), zero distortion. Document this assumption.
- Convert rotation to yaw, pitch, roll.
- Smooth with a **One Euro filter** (implement in ~40 lines, no dependency). Parameters in config.
- Neutral pose comes from the profile: features are reported relative to it.

### 4.3 Blink

- Eye openness per eye (EAR if eyelid landmarks exist).
- Blink event = openness below the profile's closed threshold for a continuous span. Record start, end, duration.
- **Voluntary long blink** vs involuntary blink: profile stores the user's involuntary blink duration distribution (from calibration). A long blink must exceed a threshold set above that distribution (for example a high percentile plus margin, chosen from data, not hardcoded).
- Both eyes must agree, otherwise the event is rejected and logged.

### 4.4 Signal quality and recovery

- Quality score from: face found, face size, landmark stability (frame-to-frame jitter), brightness mean and contrast, absolute yaw (turned away).
- Low brightness or contrast triggers CLAHE on the face ROI for subsequent frames. Log `clahe_applied`.
- If the landmark model fails for a few frames but the face was recently tracked, switch to **Lucas-Kanade optical flow** (`cv2.calcOpticalFlowPyrLK`) on the last good landmarks, for a bounded number of frames. Mark `landmark_source="flow"` and reduce quality. Never let flow-sourced frames arm a confirmation (fail closed).

### 4.5 Performance target

Measure, do not promise. Log FPS and per-stage latency. Aim for real-time on a laptop CPU at 640x480. If too slow, reduce resolution or run detection every N frames with tracking between.

---

## 5. Interaction layer

### 5.1 Gesture state machine (code decides, never the AI)

| State | Entered when | Left when |
| --- | --- | --- |
| Idle | No usable face | Face found with quality above threshold |
| Tracking | Face found, quality above threshold | Quality drops, or a gesture starts |
| Highlighting | A head tilt is counted once | Another tilt, a confirm blink, or highlight timeout |
| Armed | A long blink starts on a highlighted choice | Blink completes (go to Speaking) or is cancelled |
| Speaking | Long blink confirmed | Speech ends |
| Recovering | Signal lost mid-gesture | Signal returns, or offline list takes over |

- Pure function style: `step(state, features, profile, now) -> (new_state, events)`. Easy to unit test and replay.
- Tilt detection: displacement from neutral beyond the profile's per-direction threshold, held for a minimum duration, then return toward neutral. Debounced: one tilt counts once, with a refractory window per gesture type.
- **Undo:** a defined gesture (for example tilt down, or a double blink, decide from calibration) cancels the highlight or the armed state before speech.
- Every transition emits an event to the log.

### 5.2 Layouts

| Layout | Use when | How it works |
| --- | --- | --- |
| `four_way` | Reliable range in 4 directions | Up, down, left, right each highlight one of 4 options |
| `two_choice` | Reliable left and right only, short list | Left or right highlights one of 2 options |
| `binary_scan` | Reliable left and right only, longer list | Repeatedly halve the list: left = first half, right = second half, until one remains |

Every layout includes a **"none of these"** option that requests a narrower or different set.

### 5.3 Messages

- Default source: `config/phrasebook_en.json` (around 20 to 30 phrases, grouped: needs, feelings, people, responses).
- When cloud is available: the message agent returns 2 or 3 ranked candidates (section 7.2).
- Offline: a fixed ordered list from the phrasebook.

### 5.4 Speech

- Browser `speechSynthesis` in the UI.
- The server sends a `speak` instruction **only** when the state machine reaches Speaking from a confirmed long blink. The UI never speaks on its own.

### 5.5 UI (local web page)

- Large text, high contrast, 2 to 4 options on screen.
- Clear highlight and an "armed" indicator during the long blink (progress ring).
- Status bar: signal quality, online/offline, current layout.
- **Inspector panel** (toggle): live features, state machine state, last agent decisions with the measurements that caused them. This panel is shown in the demo video split screen.
- Keyboard shortcuts for the operator only (start session, calibrate, toggle inspector). The user never needs the keyboard.

---

## 6. Safety invariants (RULES.md + tests)

Each item becomes a line in `RULES.md` and at least one test in `tests/test_safety.py`.

1. **Code owns the state.** No agent output can change the gesture state machine state.
2. **Fail closed.** No face, low quality, flow-sourced landmarks, or a turned head means no selection and no speech.
3. **Debounce.** One physical tilt produces at most one highlight event.
4. **Confirm before speaking.** Speech happens only after Armed -> Speaking via a confirmed long blink on a highlighted option.
5. **Undo.** An undo gesture during Highlighting or Armed cancels with no speech.
6. **Stale-result guard.** Every agent request carries `session_id` and a monotonically increasing `epoch`. Responses with an older epoch are discarded and logged.
7. **Bounded agent.** Agent tool calls are validated against `config/agent_bounds.json`. Out-of-bounds or unknown tools are rejected and logged.
8. **Tool tiers enforced in code** (section 7.3). `speak_message` and `send_caregiver_alert` cannot be called by the agent at all. `access_raw_video` is forbidden and every attempt is logged.
9. **Action cap.** Maximum agent adaptation actions per session window (config). Excess is rejected.
10. **Offline fallback.** Network failure or timeout falls back to the fixed phrase list without blocking the UI.
11. **Privacy.** Only `FrameFeatures`-derived numbers leave the device, except the consented calibration clip.

---

## 7. Agent design

There are two agents. Both run on Bedrock behind Lambda, and both return only validated structured output.

### 7.1 Supervisor agent (the agentic vision loop)

Purpose: adapt the interface to the user's measured state. This is the Agentic Vision Award evidence.

**Trigger.** `monitor.py` computes a summary every N selections or T seconds, and immediately when it detects a notable change (quality drop, amplitude decline, undo spike). The summary is sent to `/supervise`.

**Input (numbers and short labels only):**

```json
{
  "session_id": "...", "epoch": 12,
  "profile_id": "...",
  "current_layout": "four_way",
  "current_params": {"tilt_hold_ms": 300, "highlight_timeout_ms": 4000, "long_blink_ms": 700},
  "window": {
    "seconds": 60,
    "selections": 5, "undos": 3, "timeouts": 2,
    "tilt_amplitude_vs_calibrated": {"left": 0.62, "right": 0.71, "up": 0.40, "down": 0.45},
    "amplitude_trend_per_min": -0.08,
    "mean_selection_time_s": 7.4,
    "quality_mean": 0.81, "quality_flags": {"low_light": 0.1},
    "long_blink_success_rate": 0.6,
    "flow_fallback_fraction": 0.05
  },
  "history": [ {"epoch": 11, "action": "adjust_timing", "outcome_after": {...}} ]
}
```

**Tools available to the supervisor:**

| Tool | Tier | Effect |
| --- | --- | --- |
| `get_window_summary(seconds)` | Read-only | Returns a summary over another window |
| `get_direction_reliability()` | Read-only | Per-direction success and false-trigger rates this session |
| `set_layout(layout)` | Reversible | Switch among `four_way`, `two_choice`, `binary_scan` |
| `adjust_timing(param, value)` | Reversible | Change a timing parameter within bounds |
| `request_recalibration(reason)` | Needs confirmation | Shows a prompt; user must accept with a confirmed gesture |
| `request_hold_still(reason)` | Reversible | Shows a non-blocking "please hold still" message |
| `no_change(reason)` | Read-only | Explicit abstention |

The supervisor runs a short Converse tool-use loop (max 3 tool calls per invocation, enforced in the Lambda). The device receives the final action list, validates each against bounds and tiers, applies them, and records the outcome.

**Why this is not a rule engine:** decisions combine several signals (amplitude per direction, trend, undo rate, quality, blink success) and the agent can query more evidence before acting. To prove value, `rule_policy.py` implements a simple deterministic baseline, and the evaluation compares both (section 9).

**Prompting rules (evidence first):**
- Base every action on specific numbers in the input. Cite them in a `reason` field.
- If quality is poor, the problem is the signal, not the user. Prefer `request_hold_still` or `no_change` over changing layout.
- Prefer the smallest reversible change. Never change more than one thing per invocation unless evidence is strong.
- If evidence is weak or conflicting, call `no_change`.

**Trace record** (written to DynamoDB and local JSONL for every invocation):

```json
{
  "trace_id": "...", "session_id": "...", "epoch": 12, "ts": "...",
  "input_summary": {...},
  "tool_calls": [{"tool": "get_direction_reliability", "result": {...}},
                 {"tool": "set_layout", "args": {"layout": "binary_scan"}}],
  "reason": "up/down amplitude at 0.40/0.45 of calibrated, 3 undos in 60 s",
  "validated": true, "applied": true, "rejected_reason": null,
  "outcome_after": {"undos": 0, "mean_selection_time_s": 5.1, "window_s": 60}
}
```

`outcome_after` is filled in later by the monitor. This is what shows OpenCV output changing a later action.

### 7.2 Message agent (suggestions)

- Endpoint `/suggest`. Input: gesture context, current quality, time of day, last spoken message, short session history (message IDs only), phrasebook.
- Output schema, validated with pydantic, anything else rejected:

```json
{"abstain": false,
 "candidates": [{"message_id": "need_water", "confidence": 0.62},
                {"message_id": "need_help", "confidence": 0.21}],
 "reason": "..."}
```

- Candidates must be IDs from the phrasebook. Free text is not allowed in this version.
- The message agent has no tools that can speak or alert.
- Two prompt variants exist for the evaluation: `careless` (context-heavy, no abstention instruction) and `evidence_first` (signal first, explicit abstention). Same model, same inputs.

### 7.3 Tool tiers (enforced in `agent/tools.py`)

| Tool | Tier | Rule |
| --- | --- | --- |
| `get_window_summary`, `get_direction_reliability`, `propose_candidates`, `no_change` | Read-only | Return data only |
| `set_layout`, `adjust_timing`, `request_hold_still`, `update_calibration_profile` | Reversible | Bounded, one-step rollback to previous value or default profile |
| `request_recalibration`, `speak_message`, `send_caregiver_alert` | Needs confirmation | Only the user's confirmed gesture unlocks it; agent cannot call speak or alert at all |
| `access_raw_video` | Forbidden | Always rejected, attempt logged |

### 7.4 Bedrock details to verify (do not hardcode guesses)

- Pick a model that supports Converse tool use and is enabled in the account region. Put the model ID in an environment variable.
- `[HUMAN]` Akshat confirms model access in the Bedrock console.
- Create a Bedrock Guardrail (denied topics off-scope, PII filters) and attach it in Converse calls. If guardrails add too much latency, measure and report it.
- Timeouts: device-side request timeout in config. On timeout, keep current settings and log.

---

## 8. Cloud (AWS CDK)

### 8.1 Resources

- **S3 bucket** with SSE-KMS (customer-managed key), block public access, lifecycle rule deleting `calibration-clips/` objects after 1 day as a backstop, and the calibrate Lambda deleting the clip immediately after processing.
- **DynamoDB** table `headvoice-audit` (PK `session_id`, SK `ts#trace_id`), on-demand billing, encryption on.
- **Lambda container images** (ECR) for `calibrate` and `replay`, both importing the shared `headvoice` package so local and cloud logic are identical. Check image size and cold start; report both.
- **Lambda (zip or container)** for `supervise`, `suggest`, `audit`.
- **API Gateway HTTP API** with routes `/calibrate`, `/replay`, `/supervise`, `/suggest`, `/audit/{session_id}`. Auth: a bearer token checked by a Lambda authorizer, token in Secrets Manager. Throttling on.
- **CloudWatch:** structured JSON logs, custom metrics (agent latency, rejections, abstentions, stale discards, cost estimate per session), and a dashboard defined in CDK.
- **IAM:** least privilege per function. Bedrock invoke only for the two agent functions.

### 8.2 Calibration flow

1. Device shows a consent screen. Without consent, use the default profile and skip upload.
2. Device records a 15 to 20 second guided clip: neutral, left, right, up, down, a few natural blinks, three deliberate long blinks.
3. Upload to S3 via a presigned URL from `/calibrate`.
4. Calibrate Lambda runs the same OpenCV 5 pipeline, computes neutral pose, per-direction reliable range, jitter, involuntary blink duration distribution, long-blink threshold, and recommended layout.
5. Lambda deletes the clip, returns the profile JSON, writes an audit record.
6. Device stores the profile locally.

**Layout choice rule** (deterministic, documented): a direction is "reliable" if its range clearly exceeds jitter by a margin and was reached consistently in the clip. 4 reliable directions = `four_way`; only left and right = `binary_scan` (or `two_choice` for short lists). Margins chosen from recorded data, written in the report.

**Stability check:** calibrate twice per volunteer and report how much thresholds and layout choice differ.

### 8.3 Replay endpoint

`/replay` accepts a session log ID (in S3) plus a config arm and returns event outputs and metrics. Recomputes pose from landmarks with OpenCV `solvePnP` in the cloud. Used by evaluation and by the judge demo.

### 8.4 Judge endpoint

A small hosted page (S3 static site behind CloudFront, or served by API Gateway) where a judge picks a sample session and an arm, runs replay, and sees the event timeline, metrics and agent trace. This satisfies "working web endpoint" without a webcam. The live webcam app is shown in the video and offered as an arranged screen-share demo.

### 8.5 Infrastructure checklist

- `[HUMAN]` AWS account, region choice, CLI credentials, CDK bootstrap.
- `cdk deploy` from a clean checkout must work using README steps.
- `cdk destroy` documented.
- Cost per session measured from CloudWatch and Bedrock token usage, reported in the report.

---

## 9. Evaluation

### 9.1 Recording protocol (`eval/protocol.md`)

`[HUMAN]` Akshat runs sessions with 4 to 6 volunteers (different faces, glasses, lighting, camera angles). Volunteers are not the target users; the report says so.

Per volunteer:
1. Consent for landmark logging (and for the calibration clip, separately).
2. Calibration twice.
3. A scripted task: select a given sequence of 10 target phrases. The script with target IDs and timestamps is saved as ground truth.
4. Repeat under one adverse condition (dim light, glasses glare, or off-angle camera).
5. One "small movements" run where the volunteer deliberately reduces head motion, approximating fatigue.

Logs saved as JSONL: `FrameFeatures` per frame, state machine events, labels. Never save raw video for evaluation.

### 9.2 Fatigue simulation (`replay/fatigue_sim.py`)

Healthy volunteers do not fatigue like target users. Simulate by progressively scaling pose displacement from neutral over the session (for example linear decay to a fraction of the original) and adding jitter. Parameters in config, reported in the report. Clearly labeled as simulated.

### 9.3 Arms

| Arm | Description |
| --- | --- |
| A. Fixed menu, generic profile | Default profile, `four_way`, no adaptation |
| B. Dwell selection | Highlight by tilt, select by holding still for a dwell time, no blink |
| C. Calibrated | User profile and calibrated layout, no adaptation |
| D. Calibrated + rule policy | Deterministic adaptation baseline |
| E. Calibrated + supervisor agent | Bedrock supervisor adaptation |

Run each arm on: normal recordings, adverse-condition recordings, small-movement recordings, and fatigue-simulated versions of normal recordings.

Note: replay of recorded features cannot capture how a user would react to a changed layout. State this limit. Live runs of arms C and E with at least two volunteers give a smaller real comparison.

### 9.4 Metrics (`eval/metrics.py`)

- **False triggers:** speech or selection events that do not match the scripted target at that time.
- **Selection time:** from options shown to confirmed selection, for correct selections.
- **Retries:** undos and timeouts per correct selection.
- **Task success rate:** correct selections / scripted targets.
- **Abstention correctness:** fraction of low-quality windows where the system correctly selected nothing.
- **Agent metrics:** latency (p50, p95), cost per session, rejected tool calls, stale discards, `no_change` rate.
- **NASA-TLX:** short form after live runs (volunteers fill in).
- **Calibration stability:** difference in thresholds and layout between two clips per person.

### 9.5 Message agent evaluation

- `eval/scenarios/`: 30 to 50 scripted scenarios with context and a ground-truth intended message ID.
- Compare: fixed order, `careless` prompt, `evidence_first` prompt.
- Metrics: top-1 and top-3 hit rate, abstention on ambiguous scenarios, invalid output rate.

### 9.6 Outputs

`run_eval.py` writes `eval/results/*.csv` and charts (matplotlib PNGs) plus a `summary.md`. All committed. Failure case clips are replays with the event timeline, not raw video.

### 9.7 Failure cases to show deliberately

Low light, head turned away, glasses glare, landmark loss with flow fallback timing out, and a case where the supervisor correctly calls `no_change`.

---

## 10. Phased tasks with acceptance criteria

Times are IST. Each phase ends with a commit and a short status note in `docs/progress.md`.

### Phase 0: Setup and verification (Oct 24, morning, ~3 h)

- [ ] Repo skeleton, `pyproject.toml`, lock file, pre-commit (ruff), pytest runs.
- [ ] `scripts/verify_opencv.py` (section 4.1). Report results.
- [ ] `scripts/get_models.sh`, `docs/models.md`.
- [ ] `[HUMAN]` AWS credentials, region, CDK bootstrap, Bedrock model access confirmed.
- [ ] `scripts/verify_bedrock.py`: one Converse call with a tool definition succeeds.

**Accept:** OpenCV 5 installed and verified, a landmark path chosen, Bedrock reachable.

### Phase 1: Vision pipeline (Oct 24, ~5 h)

- [ ] Detector, landmarks, pose with One Euro filter, blink, quality, CLAHE trigger, flow fallback.
- [ ] `scripts/live_debug.py`: webcam window overlaying landmarks, yaw/pitch/roll, eye openness, quality, FPS.
- [ ] JSONL logger for `FrameFeatures`.
- [ ] Unit tests on synthetic inputs (pose from known 3D points, blink events from synthetic openness series, filter behaviour).

**Accept:** live debug runs on Akshat's webcam; tilts and long blinks are visibly distinct in the readouts. Go/no-go (section 11).

### Phase 2: Interaction and local app (Oct 24 evening, ~5 h)

- [ ] State machine, gestures with debounce, undo, layouts, "none of these".
- [ ] Default profile, phrasebook, offline phrase list.
- [ ] FastAPI server, WebSocket, UI with highlight, armed progress, speech, status bar, inspector panel.
- [ ] Safety tests (section 6), all passing.
- [ ] `[HUMAN]` start recording volunteer sessions as soon as this works (protocol 9.1), so evaluation is not crammed at the end.

**Accept:** end-to-end offline: tilt to highlight, long blink to speak, undo cancels, poor signal never speaks.

### Phase 3: Calibration and replay (Oct 25, ~4 h)

- [ ] `calibration/analyze.py` with layout choice rule.
- [ ] Local calibration flow with consent screen.
- [ ] `replay/engine.py` producing identical events to live runs on the same logs (test with a recorded session).
- [ ] `fatigue_sim.py`.

**Accept:** calibrating changes thresholds and layout as expected; replay is deterministic.

### Phase 4: AWS deployment (Oct 25, ~5 h)

- [ ] CDK stack (section 8.1).
- [ ] Calibrate and replay Lambda containers using the shared package; clip deletion verified.
- [ ] Auth, throttling, CloudWatch dashboard.
- [ ] Device client for `/calibrate` and `/replay`, with offline fallback.
- [ ] README deploy steps tested from a clean clone.

**Accept:** `cdk deploy` works; calibration runs in the cloud and returns a profile; replay returns metrics.

### Phase 5: Agents and trace (Oct 25 night to Oct 26 morning, ~6 h)

- [ ] `monitor.py` window summaries and change detection.
- [ ] Supervisor Lambda with tool-use loop, max 3 tool calls, guardrail.
- [ ] Device-side validation, application, rollback, stale guard, action cap.
- [ ] `rule_policy.py` baseline.
- [ ] Message agent Lambda with both prompt variants and strict schema.
- [ ] Trace records to DynamoDB and local JSONL, `outcome_after` filled.
- [ ] Inspector panel shows live decisions with their reasons and measurements.
- [ ] `/audit` endpoint.

**Accept:** in a live run with deliberately reduced movement, the supervisor changes layout or timing, the trace shows the measurements that caused it, and the outcome window is recorded. A low-quality run produces `no_change` or `request_hold_still`.

### Phase 6: Evaluation (Oct 26, ~5 h)

- [ ] All arms run (section 9.3) on all recordings.
- [ ] Message agent scenarios.
- [ ] Live comparison C vs E with at least two volunteers.
- [ ] Charts, CSVs, `summary.md`, failure cases.
- [ ] Anonymized sample dataset in `data/sample/` (consented sessions only).
- [ ] Judge replay page live.

**Accept:** every metric in section 9.4 has a real number or is explicitly marked "not measured" with a reason.

### Phase 7: Documentation and video (Oct 26 evening to Oct 27, 9:00 am IST)

- [ ] `docs/architecture.md`: mermaid architecture diagram with OpenCV 5 and AWS components.
- [ ] `docs/agent_loop.md`: agent workflow diagram (perception, decision, action, verify) and one annotated real trace.
- [ ] `docs/report.md` (section 13 outline).
- [ ] `RULES.md`, README final pass.
- [ ] `docs/video_script.md` (section 14).
- [ ] `[HUMAN]` record and upload video (unlisted), submit on Devpost.
- [ ] Rotate judge credentials plan written down.

**Accept:** every item in the competition submission list (section 0) is present and linked.

---

## 11. Go/no-go checkpoint (end of Phase 1)

Test live tracking on Akshat's webcam and one other person in ordinary room lighting.

- **Go:** tilts in at least left and right, and deliberate long blinks, are reliably detected, and normal blinks do not trigger.
- **No-go:** stop and tell Akshat. Fallback options: (1) a fully binary layout using only left and right tilts with dwell confirmation instead of blinks, keeping the rest of the plan; (2) switch project to the accessibility floor-plan checker, reusing the agent gateway, tool tiers, audit log, inspector and stale guard.

Do not continue past Phase 1 without an explicit go.

---

## 12. Cut order

If time runs short, cut in this order:

1. COOL / Graviton benchmark (not planned by default; only if everything else is done).
2. Caregiver view.
3. Indian-language phrasebook and Polly voices.
4. CloudWatch dashboard polish.
5. Live C vs E comparison (keep replay comparison).
6. Message agent scenario count (reduce to 20).

**Never cut:** safety invariants and their tests, the supervisor loop and its trace, evaluation numbers with failure cases, the limitations section, the agent workflow diagram.

---

## 13. Technical report outline (`docs/report.md`)

1. Problem and users (who, the core moment, why a webcam).
2. What is new: calibration that selects the interface, vision-driven adaptation, bounded agent, honest measurement. What is not new: head and blink control, ranked suggestions with confirm-before-speak.
3. Architecture (diagram, device vs cloud split, data flow).
4. OpenCV 5 implementation (detector, landmarks, solvePnP pose, One Euro filter, blink model, quality and CLAHE, optical flow fallback, calibration analysis). Model sources and licenses.
5. Agent design (supervisor, tools, tiers, bounds, evidence-first prompting, workflow diagram, real trace).
6. AWS deployment (services, IaC, security, cost per session, latency).
7. Evaluation (protocol, arms, metrics, results tables and charts, live comparison, message agent results).
8. Failure cases.
9. Limitations: research and education prototype, not a medical device, not tested with people who rely on such a device, small volunteer group, replay cannot capture user reaction to layout changes, fatigue is simulated, sensitivity to light and angle, short calibration clip.
10. Responsible use and privacy.
11. Reproducing results (exact commands).

---

## 14. Video outline (max 5 minutes)

1. 0:00 to 0:30: team, problem, the "I need water" moment.
2. 0:30 to 1:30: live demo, split screen: user view left, inspector panel right. Tilt, highlight, long blink, speech. Undo.
3. 1:30 to 2:15: calibration choosing a layout for a user with limited range.
4. 2:15 to 3:00: supervisor adapting when movement shrinks, with the trace on screen; then a low-light case where it correctly does nothing.
5. 3:00 to 3:45: architecture and AWS, privacy (video stays on device).
6. 3:45 to 4:40: results: arms comparison, failure cases, limits.
7. 4:40 to 5:00: what is next, "research prototype, not a medical device". State that a volunteer plays the user.

---

## 15. Definition of done

- [ ] Fresh clone + README steps: local app runs offline with default profile.
- [ ] `pytest` passes, including all safety tests.
- [ ] `cdk deploy` works and the judge replay page is reachable.
- [ ] At least one real trace where OpenCV measurements changed a later supervisor action, with outcome recorded.
- [ ] Evaluation results committed, with failure cases and limitations.
- [ ] Report, RULES.md, architecture diagram, agent workflow diagram, video link all present.
- [ ] No raw video in the repo or in S3 after processing.
- [ ] Submitted on Devpost before Oct 27, 12:15 pm IST.
