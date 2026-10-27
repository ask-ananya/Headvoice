# Video script (under 5 minutes)

Recording setup: screen capture at full screen, browser speech audible (system volume up), quiet room, ordinary room light. The developer plays the user and says so on camera. Start the app with `.\scripts\run_app.ps1` so the agent is on. Use only real behaviour; do not stage the helper panel.

## 0:00 to 0:25: the problem

On screen: the HeadVoice landing page (website), then the developer in front of the laptop.

Narration: "Some people cannot speak clearly or use their hands, but can still move their head a little and blink. HeadVoice turns an ordinary webcam into a voice. I'm the developer, not a target user; I'm showing how it works."

## 0:25 to 0:55: setup

1. Open http://localhost:8765, click "Set up HeadVoice".
2. Camera check: the live preview, the three ticks (camera working, face found, resting position learned). Point at the camera picker: "Frozen phone-webcam feeds are flagged."
3. Practice: say two messages.

Narration: "Setup checks the camera, learns your resting position in a second, and lets you practice. Everything here runs on this computer; the camera picture never leaves it."

## 0:55 to 1:45: talking

On the message screen; the camera window is visible in the top-right corner.

1. Tilt left: "I need help" highlights. Narration: "A head tilt highlights a message. OpenCV 5 finds the face with YuNet, fits 68 landmarks, and measures head pose with solvePnP."
2. Hold a long blink: the ring around the card fills, the app speaks. Narration: "A long blink confirms. Eyelid landmarks barely move, so the eye state comes from a small OpenCV classifier on the eye images: HOG features and an SVM."
3. Tilt up, then two quick blinks: cancelled. Narration: "Two quick blinks cancel. Nothing is spoken without my own confirming blink."
4. Turn away from the screen while a message is highlighted: nothing is spoken, the guidance strip says "Face the screen". Narration: "If the signal is poor, it fails closed."
5. Open My messages, add "Turn on the cricket", go back: it is first. Narration: "Helpers can add the person's own phrases. They stay on this computer."

## 1:45 to 2:20: calibration chooses the interface

1. Press `C`, show the consent screen, press `L` (on this computer).
2. Follow the prompts, keeping up and down movement small.
3. Result notice: left and right scanning. Narration: "Calibration measures how far I can tilt against my own jitter, and how I blink. Up and down were not reliable, so it picked a left and right scanning layout with personal thresholds."

## 2:20 to 3:10: the agent adapts, and knows when not to

Open the helper panel (`I`). Keep using the app for a few minutes beforehand so the first minute's warm-up has passed, then make tilts steadily smaller.

1. Near misses rise; an adaptation decision appears in the helper panel: `adjust_sensitivity`, with the numbers it cited, and a minute later its outcome. Narration: "A supervisor agent on Amazon Bedrock reads only numbers from OpenCV: tilt size against my range, near misses, quality. It can check more evidence first, then makes one small, reversible change. My computer validates it and measures the result a minute later."
2. Dim the light or look away briefly: "Please hold still" appears. Narration: "When the problem is the signal, it asks me to hold still instead of changing the interface."
3. Mention the guards on screen: "It can't change anything in the first minute, can't go past fixed limits, and if a helper picks a layout, it stays."

If the live take does not produce a clean decision, show the real trace from `docs/agent_loop.md` (session `developer_1027_044434_8min`) and say it is from an earlier session.

## 3:10 to 3:50: architecture and AWS

Show the system diagram in `docs/architecture.md`. Narration: "The camera, the OpenCV pipeline, the gesture logic and speech run on the device. AWS runs the Bedrock agents behind a guardrail, calibration and replay in Lambda with the same OpenCV 5 code, an encrypted S3 bucket, a DynamoDB audit trail, a CloudWatch dashboard and the website, all from one CDK stack. Only numbers cross the boundary."

Show the judge replay page: pick `developer_1027_044434_8min`, "Compare all setups", scroll to the agent decisions.

## 3:50 to 4:40: results and honesty

Show the results tables from `docs/report.md`.

Narration (numbers as in the report):

- "We had no volunteers. Real data is my own: 13 sessions and 30 agent decisions on Bedrock, with the measured outcome of each change."
- "To compare setups we built a simulated-users study, clearly labeled as simulation: 30 simulated users driving the real app. Calibration took users with limited up-and-down movement from 49% to 100% task success; under simulated fatigue, adaptation lifted success from 59% to 78%."
- "We report what went wrong: the agent sometimes flipped layouts back and forth, and changed settings when it should only have asked me to hold still. The device caught a request that went past its limits."

## 4:40 to 5:00: close

Narration: "HeadVoice is a research prototype, not a medical device. Next: testing with people who rely on such tools and their care teams, and fixing the agent's layout flip-flopping. Thank you."
