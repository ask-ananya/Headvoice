# HeadVoice architecture

HeadVoice is split by one rule: **the camera picture and every decision about what to say stay on the user's computer.** The device runs the webcam, the OpenCV 5 pipeline, the gesture state machine, the user interface and speech. AWS runs what benefits from the cloud: the adaptation and message agents on Amazon Bedrock, calibration and replay jobs (the same OpenCV 5 code in a container), the audit trail, the public website and the judge replay page. Only numeric summaries cross that boundary, plus a calibration clip the user explicitly agrees to upload, which is deleted as soon as it is read.

Contents:

1. [System overview](#1-system-overview)
2. [Per-frame vision pipeline](#2-per-frame-vision-pipeline)
3. [Gesture state machine](#3-gesture-state-machine)
4. [Local app](#4-local-app)
5. [Adaptation loop](#5-adaptation-loop)
6. [Calibration](#6-calibration)
7. [Replay, evaluation and the judge page](#7-replay-evaluation-and-the-judge-page)
8. [AWS deployment](#aws-deployment)
9. [Data, privacy and security](#9-data-privacy-and-security)
10. [Code map](#10-code-map)

---

## 1. System overview

```mermaid
flowchart LR
  subgraph Device["User's computer (Python, localhost:8765)"]
    CAM["Webcam"] --> PIPE["OpenCV 5 vision pipeline<br/>FrameFeatures at about 30 fps"]
    PIPE --> SM["Gesture state machine<br/>(pure function, code decides)"]
    SM --> UI["Web UI<br/>landing, setup, message screen,<br/>My messages, helper panel"]
    UI --> TTS["Browser speech"]
    PIPE --> MON["Session monitor"]
    SM --> MON
    MON <--> ADP["Adapter<br/>validate, apply, measure outcome"]
    ADP -->|"bounded profile changes"| SM
    SM --> LOGS[("Local logs<br/>sessions, traces,<br/>custom messages")]
  end

  subgraph AWS["AWS, one CDK stack (us-east-1)"]
    API["API Gateway HTTP API<br/>Lambda token authorizer, throttling"]
    LAMBDA["Lambda functions<br/>(one OpenCV 5 container image)"]
    BR["Amazon Bedrock<br/>Claude Haiku 4.5 + Guardrail"]
    S3[("S3, KMS encrypted")]
    DDB[("DynamoDB<br/>audit trail")]
    CW["CloudWatch<br/>logs, metrics, dashboard"]
    CF["CloudFront website"]
    API --> LAMBDA
    LAMBDA --> BR
    LAMBDA --> S3
    LAMBDA --> DDB
    LAMBDA -.-> CW
  end

  ADP -->|"window summaries (numbers only)"| API
  API -->|"one tool call"| ADP
  ADP -->|"decision traces"| API
  UI -.->|"calibration clip, only with consent"| API
  JUDGE["Judge (browser)"] --> API
  VISITOR["Visitor (browser)"] --> CF
  CF -.->|"Windows download, runs locally"| Device
```

| Part | Runs on | Why there |
| --- | --- | --- |
| Camera, OpenCV pipeline | Device | Real time, and the picture never leaves the computer |
| Gesture state machine, speech | Device | Safety: no network or model can make the app speak |
| Web UI, camera window, custom messages | Device (FastAPI, localhost only) | The preview and personal phrases stay local |
| Offline phrase list and rule policy | Device | Works with no connection |
| Session monitor and adapter | Device | Every agent decision is validated where it is applied |
| Supervisor and message agents | Lambda + Bedrock | Model access, guardrail, central audit |
| Calibration analysis | Device, and Lambda for consented clips | Same code in both places |
| Replay and evaluation | Lambda and locally | Judges can test without a webcam |
| Audit trail | DynamoDB | Every invocation and device trace, per session |
| Website | CloudFront + private S3 | Public information and the download |

---

## 2. Per-frame vision pipeline

Every camera frame becomes one `FrameFeatures` record (`src/headvoice/vision/pipeline.py`): time, face found, landmarks, landmark source, yaw, pitch, roll relative to the user's neutral, left and right eye openness, quality score and quality flags.

```mermaid
flowchart TD
  F["Camera frame, 640x480 BGR"] --> FZ{"Frozen feed?<br/>10 identical frames"}
  FZ -->|yes| BADQ["quality 0, flag frozen_feed"]
  FZ -->|no| CL{"Face dark or flat?<br/>(hysteresis)"}
  CL -->|yes| CLAHE["cv2.createCLAHE<br/>on the face region"]
  CL -->|no| DET
  CLAHE --> DET["cv2.FaceDetectorYN<br/>YuNet 2026may, score 0.6"]
  DET -->|face| LBF["cv2.face FacemarkLBF<br/>68 landmarks"]
  DET -->|missed| BOX{"Recent landmarks?<br/>up to 15 frames"}
  BOX -->|yes| LBF2["LBF on a box carried from<br/>the previous landmarks"]
  BOX -->|no| FLOW["cv2.calcOpticalFlowPyrLK<br/>up to 10 frames, source flow"]
  LBF --> POSE
  LBF2 --> POSE
  FLOW --> POSE
  POSE["Pitch, roll: cv2.solvePnP (6 points)<br/>Yaw: nose offset along the eye line"] --> EURO["One Euro filter"]
  EURO --> NEU["Relative to neutral<br/>(auto-learned in 1 s, follows slow drift)"]
  LBF --> EYE["Eye crops aligned to the eye corners<br/>cv2.HOGDescriptor + cv2.ml.SVM"]
  EYE --> MED["3-frame median<br/>openness: 1 open, 0 closed"]
  NEU --> QUAL["Quality: face size, light, contrast,<br/>stability, turned away, source"]
  MED --> FF["FrameFeatures"]
  QUAL --> FF
  BADQ --> FF
```

Measured on a laptop CPU: detection 10 to 11 ms, landmarks 4.5 ms, pose 0.6 ms, eye state 0.17 ms; the 30 fps camera is the limit. Why each choice was made (YuNet export, LBF over unavailable alternatives, geometric yaw, the eye-state model) is documented with measurements in [models.md](models.md) and the [report](report.md#4-opencv-5-implementation).

---

## 3. Gesture state machine

`step(state, features, profile, now) -> (state, events)` in `src/headvoice/interaction/state_machine.py`. It is a pure function: no network, no model, and the agent layer never imports it. Default timings from `config/default_profile.json`.

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> Tracking: usable frame
  Tracking --> Idle: unusable for 300 ms
  Tracking --> Tracking: long blink, nothing highlighted / show other messages
  Tracking --> Highlighting: tilt held 300 ms past threshold
  Highlighting --> Highlighting: another tilt / retarget
  Highlighting --> Armed: both eyes close on one highlighted message
  Armed --> Speaking: eyes stay closed for long_blink_ms (700 ms)
  Armed --> Highlighting: eyes open too early
  Highlighting --> Tracking: undo (double blink) or 4 s timeout
  Armed --> Tracking: undo
  Highlighting --> Recovering: unusable frame
  Armed --> Recovering: unusable frame (arm cancelled)
  Recovering --> Highlighting: signal back, highlight kept
  Recovering --> Idle: 3 s without signal
  Speaking --> Tracking: speech ended (or 6 s timeout)
```

A frame is usable only if a face is found, landmarks come from the model (never optical flow), the head is not turned away, quality is at least 0.5 and pose is known. Every unusable frame cancels an armed blink, and after a dropout or speech the eyes must be seen open before a new blink counts. One function, `_confirm`, is the only place a speak event exists.

---

## 4. Local app

```mermaid
flowchart LR
  subgraph Browser["Browser (localhost:8765)"]
    L["/ landing"] --> ST["/start setup:<br/>welcome, camera check,<br/>calibration, practice"]
    ST --> AP["/app message screen<br/>camera window top right,<br/>helper panel"]
    L --> AP
    AP --> MS["/messages<br/>My messages"]
  end
  subgraph Server["FastAPI server"]
    WS["WebSocket /ws<br/>snapshots 15 per second,<br/>speak and notice messages"]
    REST["REST: /api/state, /api/cameras,<br/>/api/camera, /api/messages,<br/>/api/condition"]
    MJPG["/preview.mjpg<br/>mirrored, annotated"]
    CTRL["Controller<br/>state machine, menu, calibration,<br/>tasks, adapter, logging"]
    CAMT["Camera thread<br/>pipeline per frame"]
    CAMT --> CTRL
    WS <--> CTRL
    REST <--> CTRL
    CAMT --> MJPG
  end
  AP <--> WS
  ST <--> WS
  ST --> REST
  MS --> REST
  AP --> MJPG
```

The browser speaks only when the server sends a `speak` message, which the controller sends only for a speak event from the state machine. The same `Controller` class drives the replay engine, so replayed sessions produce the same events as live ones.

---

## 5. Adaptation loop

The adapter (`src/headvoice/agent/adapter.py`) runs perceive, decide, act, verify. With policy `agent` the decision is made by the supervisor on Bedrock; with policy `rule` by the deterministic baseline on the device.

```mermaid
sequenceDiagram
  autonumber
  participant P as OpenCV pipeline
  participant M as Session monitor
  participant A as Adapter (device)
  participant G as API Gateway + supervise Lambda
  participant B as Bedrock (Haiku 4.5 + guardrail)
  participant D as DynamoDB audit

  P->>M: FrameFeatures every frame
  Note over M: deliberate tilt attempts and near misses,<br/>amplitude vs calibrated range, trend,<br/>undos, timeouts, quality
  M->>A: trigger (quality drop, amplitude decline,<br/>undo spike, 5 selections, 60 s)
  Note over A: skipped in the first minute (except quality drop)<br/>and while the last change awaits its outcome
  A->>A: build request, privacy check (numbers only)
  A->>G: window summary + params + last 3 decisions with outcomes
  G->>B: Converse with tools (max 3 tool calls)
  B-->>G: optional read-only call, e.g. get_direction_reliability
  G-->>B: answer from the evidence pack
  B-->>G: one action, e.g. adjust_sensitivity 0.8
  G->>D: invocation record
  G-->>A: action + reason + tool calls
  A->>A: validate: tier, bounds, epoch, action cap, layout lock
  alt valid
    A->>A: apply to profile, keep one-step rollback
  else invalid
    A->>A: reject and log the reason
  end
  A->>D: trace record
  Note over A,M: 60 s later
  M->>A: outcome window summary
  A->>D: trace updated with outcome_after
```

Tool tiers, bounds and their tests: [agent_loop.md](agent_loop.md). A real trace from a live session is in the same document.

---

## 6. Calibration

The guided clip (`src/headvoice/calibration/protocol.py`, about 25 s) asks for neutral, each tilt direction, natural blinks and three long blinks. The analysis (`analyze.py`) is the same on the device and in the calibrate Lambda.

```mermaid
sequenceDiagram
  autonumber
  participant U as User and helper
  participant App as Local app
  participant API as API Gateway
  participant L as calibrate Lambda
  participant S3 as S3 calibration-clips/

  U->>App: press C, consent screen
  alt On this computer (L)
    App->>App: record FrameFeatures with step labels
    App->>App: analyze(): neutral, per-direction range vs jitter,<br/>layout, thresholds, blink lengths, eye levels
  else Secure cloud processing (Y)
    App->>App: record clip + timestamps locally
    App->>API: request upload URL
    API->>L: upload_url
    L-->>App: presigned PUT (5 minutes)
    App->>S3: upload clip (KMS encrypted)
    App->>App: delete local clip
    App->>API: analyze key + timestamps
    API->>L: analyze
    L->>S3: download once
    L->>S3: delete clip immediately
    L->>L: same OpenCV 5 pipeline + analyze()
    L-->>App: profile + report
  end
  App->>App: apply profile, save to data/profiles/
```

Layout rule: four reliable directions give `four_way`; reliable left and right only give `binary_scan` (left/right scanning); otherwise the current layout is kept and the result is flagged. A direction is reliable if its range clears both 8 degrees and six times the user's jitter, and was reached consistently. A 1-day lifecycle rule on `calibration-clips/` is a backstop if the Lambda ever fails before deleting.

---

## 7. Replay, evaluation and the judge page

```mermaid
flowchart TD
  subgraph Sources["Session sources"]
    LIVE["Live sessions<br/>data/sessions/*.jsonl"] --> ANON["make_sample.py<br/>strip landmarks and face boxes"]
    SIM["Simulated users<br/>sim_users.py --export-samples<br/>(labeled simulated)"]
  end
  ANON --> SAMPLE[("S3 sample/")]
  SIM --> SAMPLE
  JUDGE["Judge page<br/>/judge"] -->|"list"| JL["judge_datasets Lambda"]
  JL --> SAMPLE
  JUDGE -->|"session, setup A to E, fatigue"| JR["judge_replay Lambda"]
  JR --> CACHE{"Setup E cached?<br/>S3 cache/, keyed by ETag"}
  CACHE -->|yes| OUT["Metrics, event timeline,<br/>agent decisions with evidence"]
  CACHE -->|no| RUN["Replay through the real Controller<br/>(optional fatigue simulation)"]
  RUN -->|"setup E"| BR["Bedrock supervisor"]
  RUN --> OUT
  WARM["warm_replay_cache.py<br/>direct Lambda invoke, 5 min timeout"] --> JR
```

- Setups: A fixed menu with generic profile, B dwell selection, C calibrated, D calibrated + rule policy, E calibrated + supervisor agent (`src/headvoice/replay/arms.py`).
- Agent replays of long sessions take longer than API Gateway's 30 s limit, so they are precomputed and cached in S3.
- The simulated-users study (`eval/sim_users.py`) runs closed loop against the real Controller: each simulated user reads the screen, plans tilts and blinks, and is calibrated by the real analysis. Replay of logged sessions is open loop and cannot show how a person would react to a changed layout.

---

## AWS deployment

One CDK v2 stack, `infra/headvoice_stack.py` (87 resources).

```mermaid
flowchart TB
  subgraph Public["Public"]
    CF["CloudFront<br/>HTTPS, security headers"] --> SITEB[("S3 site bucket<br/>private, origin access control")]
    JP["GET /judge, /judge/datasets<br/>POST /judge/replay<br/>(no token, sample data only, 0.5 req/s)"]
  end

  subgraph Api["API Gateway HTTP API (throttled 10 req/s, burst 20)"]
    AUTH["Lambda authorizer<br/>bearer token from Secrets Manager"]
    R1["POST /supervise"]
    R2["POST /suggest"]
    R3["POST /calibrate"]
    R4["POST /replay"]
    R5["POST /audit, GET /audit/{session_id}"]
    AUTH --- R1 & R2 & R3 & R4 & R5
  end

  subgraph Functions["Lambda (one container image: Python 3.11, OpenCV 5 headless contrib, models)"]
    F1["supervise"]
    F2["suggest"]
    F3["calibrate (2 GB)"]
    F4["replay (2 GB)"]
    F5["audit"]
    F6["judge page, datasets, replay"]
  end

  R1 --> F1
  R2 --> F2
  R3 --> F3
  R4 --> F4
  R5 --> F5
  JP --> F6

  F1 & F2 & F4 & F6 --> BR["Bedrock: Claude Haiku 4.5<br/>inference profile + Guardrail<br/>(content filters, prompt attacks,<br/>PII anonymization, medical-advice topic)"]
  F3 --> S3D[("S3 data bucket, KMS key with rotation<br/>calibration-clips/ (1-day expiry)<br/>datasets/, sample/, cache/")]
  F4 --> S3D
  F6 --> S3D
  F1 & F3 & F5 --> DDB[("DynamoDB headvoice-audit<br/>session_id + sort key, on demand")]
  F1 & F2 & F3 & F4 & F5 & F6 -.-> CW["CloudWatch: 2-week logs,<br/>embedded metrics, HeadVoice dashboard"]
```

| Concern | How it is handled |
| --- | --- |
| Authentication | Bearer token checked by a Lambda authorizer (constant-time compare), stored in Secrets Manager, results cached 5 minutes |
| Least privilege | Only the agent and replay functions may call Bedrock; only calibrate may touch `calibration-clips/`; replay and judge functions read `sample/` and read/write `cache/` |
| Encryption | Data bucket: customer-managed KMS key with rotation, TLS enforced, public access blocked; DynamoDB encrypted; site bucket private behind CloudFront |
| Throttling | Stage default 10 requests per second (burst 20); public judge replay 0.5 per second (burst 2) |
| Observability | Embedded metric format: agent latency, tool calls, no-change rate, estimated cost, device-side rejections, stale discards, calibration time; Lambda errors; one dashboard |
| One image for all functions | Cloud and device run identical code; numpy is pinned to 2.2.6 on both because newer wheels need a newer glibc than the Amazon Linux 2 base |
| Teardown | `cdk destroy` removes buckets (auto-delete), table, key and log groups |

Measured: cold start 430 to 620 ms; supervisor decision about 3 to 4 s end to end; replay of a 7-minute session through setups A to D under 3 s.

---

## 9. Data, privacy and security

```mermaid
flowchart LR
  subgraph Never["Never leaves the computer"]
    V["Camera frames"]
    PV["Camera preview"]
    LM["Landmark positions<br/>(local logs only)"]
    CM["My messages"]
  end
  subgraph Numbers["May go to AWS (checked by io/privacy.check_outbound)"]
    WS["Window summaries:<br/>tilt sizes, near misses,<br/>quality, counts, params"]
    TR["Decision traces"]
  end
  subgraph Consent["Only with explicit consent"]
    CLIP["25 s calibration clip<br/>encrypted, read once, deleted"]
  end
  subgraph Shared["Published for judges"]
    SMP["Anonymized samples:<br/>angles, eye openness, quality, events<br/>(no landmarks, no face boxes)"]
  end
```

- The outbound check rejects binary data, arrays, landmark or frame fields, long lists and long strings; the supervise Lambda runs it again.
- The agent cannot speak, send alerts or access video: those tools are refused in code on the device.
- The eye-state model file stores only the linear SVM weight vector; OpenCV's copy of the training vectors (features of the developer's eye crops) was removed.

---

## 10. Code map

| Path | Responsibility |
| --- | --- |
| `src/headvoice/vision/` | Camera opening and liveness, face detection, landmarks, pose, eye state, blink helpers, quality and CLAHE, optical-flow fallback, the pipeline |
| `src/headvoice/interaction/` | Gesture state machine, tilt tracking and debounce, layouts, menu paging |
| `src/headvoice/calibration/` | Guided protocol, analysis, recording and clip processing |
| `src/headvoice/agent/` | Session monitor, rule policy, supervisor and message agents, tool tiers and bounds, adapter loop, trace records |
| `src/headvoice/replay/` | Replay engine (drives the real Controller), fatigue simulation, arms, metrics |
| `src/headvoice/io/` | Schemas, JSONL logs, privacy check, cloud client, offline fallback, custom messages |
| `src/headvoice/app/` | FastAPI server, Controller, web UI |
| `src/headvoice/cloud/` | Lambda handlers, judge page |
| `infra/` | CDK app and stack |
| `cloud/` | Lambda Dockerfile and pinned requirements |
| `web/` | Website pages and the built site |
| `eval/` | Protocol, scenarios, evaluation runners, simulated users, results |

See also: [agent_loop.md](agent_loop.md) (agent workflow and traces), [report.md](report.md) (technical report), [models.md](models.md) (models and their evaluation), [../RULES.md](../RULES.md) (safety rules).
