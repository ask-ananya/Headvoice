# One image for every HeadVoice Lambda. The repo layout is mirrored under /var/task so the
# package finds config/ and models/ exactly as on the device.
FROM public.ecr.aws/lambda/python:3.11

COPY cloud/requirements-lambda.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY config/ /var/task/config/
COPY models/face_detection_yunet_2026may.onnx models/lbfmodel.yaml models/eye_state_svm.yml models/eye_state.json /var/task/models/
COPY src/ /var/task/src/

ENV PYTHONPATH=/var/task/src
# Each function overrides this with its own handler.
CMD ["headvoice.cloud.handlers.supervise"]
