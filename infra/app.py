"""CDK app entry point: cdk deploy / cdk destroy."""

import os

import aws_cdk as cdk
from headvoice_stack import HeadVoiceStack

app = cdk.App()
HeadVoiceStack(
    app,
    "HeadVoice",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
)
app.synth()
