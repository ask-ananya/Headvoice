"""HeadVoice: head tilts to choose, a long blink to confirm, the app speaks."""

# Session log format. 2 = eye openness from the eye-state model (about 1 open, 0 closed)
# and yaw from 2D landmark geometry. Format 1 logs (EAR openness, PnP yaw) are not
# comparable and are excluded from evaluation.
LOG_FORMAT = 2
