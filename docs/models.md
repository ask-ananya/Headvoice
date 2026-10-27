# Models

YuNet and LBF are downloaded by `python scripts/get_models.py` (or `sh scripts/get_models.sh`, or automatically by `Start HeadVoice.bat`) into `models/`. The eye-state model is trained in this project and ships with the app and the Lambda image.

| File | Purpose | Source | License | SHA-256 |
| --- | --- | --- | --- | --- |
| `face_detection_yunet_2026may.onnx` | Face detection (`cv2.FaceDetectorYN`) | [opencv_zoo/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT (Shiqi Yu) | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` |
| `eye_state_svm.yml`, `eye_state.json` | Eye open/closed from eye-region appearance (HOG + linear `cv2.ml` SVM) | Trained in this project by `scripts/train_eye_model.py` | Project's own | regenerate with the script |
| `lbfmodel.yaml` | 68-point facial landmarks (`cv2.face.createFacemarkLBF`) | [kurnianggoro/GSOC2017](https://github.com/kurnianggoro/GSOC2017/tree/master/data), trained for the OpenCV contrib Facemark API (GSoC 2017) | No license file in the source repository. Distributed as the reference model for OpenCV's `face` contrib module. Treat as research use only; noted as a limitation. | `70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b` |

## Why these

- **YuNet 2026may**, not 2023mar: the opencv_zoo README states the OpenCV 5.x ONNX Runtime engine needs dynamic input dims, which only the 2026may export has.
- **Landmark path**: the plan listed three candidates. Path 1 (an ONNX facial landmark model via `cv2.dnn`) was not used because opencv_zoo ships no facial landmark model. Path 2 (`FacemarkLBF`) works in the `opencv-contrib-python==5.0.0.93` build and gives the iBUG 68-point layout used for pose, eye crops and yaw. Its eyelid points turned out too stable for blink detection (see below), so blinks use the eye-state model on eye crops located by the landmarks.
- **Yaw**: 6-point `solvePnP` with a generic face gave two near-equal solutions on real data; yaw is computed from the nose offset along the eye line (`vision/pose.py::yaw_from_landmarks`), while pitch and roll stay on `solvePnP`.

## Verified (scripts/verify_opencv.py, Oct 25 2026)

OpenCV 5.0.0 on Python 3.11. On the OpenCV sample image: YuNet found 1 face in 12 ms, LBF fitted 68 landmarks in 4.5 ms, EAR right 0.41, left 0.44.

## Eye-state model (blink detection)

Why: LBF eyelid landmarks barely move when the eyes close. On labeled recordings, eye aspect ratio (EAR) separated open from closed eyes at chance level (AUC 0.34 to 0.75; 0.5 is chance), and in live use open eyes drifted to the closed threshold.

What: each eye is cropped along its corner line (48x24 grayscale, the user's left eye mirrored), described with `cv2.HOGDescriptor`, and scored with a linear `cv2.ml.SVM`. The score is mapped to openness (about 1.0 open, 0.0 closed). Cost: 0.17 ms per frame for both eyes.

Training data: three guided recordings of the developer (two with glasses, one without, about 63 s each, 6,054 labeled eye crops: eyes open, closed, looking down, tilted, squinting). Only eye crops were stored, with consent, and they were deleted after training. The model is therefore fitted to one person. Per-user calibration measures the user's own open and closed levels and moves the thresholds (close at 45% of the gap above the closed level, reopen at 70%); it does not retrain the model.

Stored file: `eye_state_svm.yml` keeps only the linear SVM's single weight vector (6.3 KB). OpenCV's `save()` also writes the uncompressed support vectors (1,325 HOG vectors of individual training crops, 7.6 MB); `scripts/train_eye_model.py` removes them with `save_compact()` because they are not needed for prediction and are derived from the developer's eye images. Predictions are identical (checked on 500 inputs).

Held-out evaluation (train on two recordings, test on the third, thresholds close below 0.6 and reopen above 0.8, both eyes required):

| Held-out recording | Long blinks detected (700 ms) | Open prompts with a false long blink | Longest run during natural blinking |
| --- | --- | --- | --- |
| Glasses 1 | 8/8 | 2/9 (looking down) | 6 frames |
| Glasses 2 | 8/8 | 0/9 | 2 frames |
| No glasses | 8/8 | 2/9 (squint, looking down) | 14 frames (17 needed) |

Known failure cases: looking down with the eyes (lids drop) and deliberate squinting. Head-down pitch is handled in the state machine (eye closure ignored while the head is pitched down); eye-only downward gaze is not detected.
