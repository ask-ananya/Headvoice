# Recording protocol: single participant (the developer)

## What was actually recorded

13 sessions (22.5 minutes) in the normal condition with the agent on, mostly free use, published anonymized on the judge page as `developer_*`; one contains the real agent trace (`docs/agent_loop.md`). Scripted tasks were run only briefly (3 short tasks), and the dim-light, small-movement, glare and second-calibration sessions below were not recorded. The report therefore uses the live sessions for traces and behaviour, and the simulated-users study for setup comparisons.

The full protocol below remains the plan for further recordings.

There are no volunteers. The real evaluation data comes from one participant, the developer, a healthy adult who is not a target user. The report states n=1 everywhere these numbers appear. A separate simulated-users study (`eval/sim_users.py`) compares interface setups across user types; it is labeled as simulation and is never mixed with these recordings.

Nothing here records video. Session logs hold numbers only (pose angles, eye openness, quality, landmarks, events), in `data/sessions/`.

## Setup (once)

1. Start the app connected to AWS with adaptation on: `.\scripts\run_app.ps1` (agent policy, camera 1).
2. Open http://localhost:8765, go through Set up (camera check, practice).
3. Sit about 60 cm from the camera, face centered, normal room light.

## Sessions (about 40 minutes in total)

Use the helper panel (gear button or key `I`), Research section, to set the condition before each session. Press `S` to start, `T` to start the scripted 10-message task, `S` again to stop when the task is done.

| # | Condition (Research panel) | What to do | Purpose |
| --- | --- | --- | --- |
| 1 | Normal | Calibrate (`C`, then `L`), then a scripted task | Baseline, calibration 1 |
| 2 | Normal | Calibrate again, then a scripted task | Calibration stability (compare the two reports) |
| 3 | Dim light | Turn off the main light, keep a dim lamp, scripted task | Adverse condition |
| 4 | Small movements | Scripted task with deliberately small tilts | Approximates limited range |
| 5 | Small movements | 3 to 4 minutes, start normally and make tilts steadily smaller; keep going even when they stop counting | The real agent trace: OpenCV measurements change a later supervisor action, with its outcome |
| 6 | Glasses glare (optional) | Scripted task with a bright screen or lamp reflecting in glasses | Failure case |
| 7 | Normal (optional) | Look away from the screen and back during a highlight, cover the camera once | Failure cases: turned away, signal lost |

After each session, nothing else is needed: logs and agent traces are written automatically (`session_*.jsonl`, `traces_*.jsonl`).

## Afterwards

```sh
uv run python eval/run_eval.py --agent api     # arms A to E on these recordings
```

The real trace for `docs/agent_loop.md` comes from session 5's `traces_*.jsonl`: pick the first applied adaptation with an `outcome_after`.

To share a session on the judge page (anonymized, landmarks removed), only with the participant's agreement:

```sh
uv run python scripts/make_sample.py data/sessions/<session>.jsonl --name developer_normal
aws s3 cp data/sample/developer_normal.jsonl s3://<BucketName>/sample/
```
