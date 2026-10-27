"""HeadVoice AWS stack (CDK v2). See plan section 8 and docs/architecture.md."""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from aws_cdk import aws_secretsmanager as sm
from constructs import Construct

REPO = Path(__file__).resolve().parent.parent
MODEL_PROFILE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class HeadVoiceStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)

        # Storage ------------------------------------------------------------------------
        key = kms.Key(
            self,
            "DataKey",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY,
            description="HeadVoice data at rest",
        )
        bucket = s3.Bucket(
            self,
            "Data",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=key,
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                # Backstop: the calibrate Lambda deletes each clip right after reading it.
                s3.LifecycleRule(prefix="calibration-clips/", expiration=Duration.days(1)),
            ],
        )
        table = ddb.Table(
            self,
            "Audit",
            table_name="headvoice-audit",
            partition_key=ddb.Attribute(name="session_id", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="sk", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            encryption=ddb.TableEncryption.AWS_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
        )
        token = sm.Secret(
            self,
            "ApiToken",
            description="Bearer token for the HeadVoice API",
            generate_secret_string=sm.SecretStringGenerator(
                exclude_punctuation=True, password_length=40
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Bedrock guardrail: off-scope topics denied, PII anonymized, prompt attacks blocked.
        guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name="headvoice-guardrail",
            blocked_input_messaging="Request blocked by the HeadVoice guardrail.",
            blocked_outputs_messaging="Response blocked by the HeadVoice guardrail.",
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type=t, input_strength="MEDIUM", output_strength="MEDIUM"
                    )
                    for t in ("HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT")
                ]
                + [
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="PROMPT_ATTACK", input_strength="HIGH", output_strength="NONE"
                    )
                ]
            ),
            sensitive_information_policy_config=bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                pii_entities_config=[
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(type=t, action="ANONYMIZE")
                    for t in ("EMAIL", "PHONE", "NAME", "ADDRESS")
                ]
            ),
            topic_policy_config=bedrock.CfnGuardrail.TopicPolicyConfigProperty(
                topics_config=[
                    bedrock.CfnGuardrail.TopicConfigProperty(
                        name="MedicalAdvice",
                        type="DENY",
                        definition=(
                            "Diagnosing conditions, recommending treatments or medication doses."
                        ),
                        examples=[
                            "What medication should the user take?",
                            "Diagnose the user's condition.",
                        ],
                    )
                ]
            ),
        )
        guardrail_version = bedrock.CfnGuardrailVersion(
            self, "GuardrailV1", guardrail_identifier=guardrail.attr_guardrail_id
        )

        # Lambdas: one image, one handler each ---------------------------------------------
        env = {
            "BUCKET": bucket.bucket_name,
            "TABLE": table.table_name,
            "TOKEN_SECRET_ARN": token.secret_arn,
            "MODEL_ID": MODEL_PROFILE,
            "GUARDRAIL_ID": guardrail.attr_guardrail_id,
            "GUARDRAIL_VERSION": guardrail_version.attr_version,
            # Haiku 4.5 on-demand list price (USD per million tokens); confirm on the AWS
            # Bedrock pricing page before quoting costs.
            "PRICE_IN_PER_MTOK": "1.0",
            "PRICE_OUT_PER_MTOK": "5.0",
        }

        def fn(name: str, handler: str, memory: int, timeout_s: int) -> lambda_.DockerImageFunction:
            return lambda_.DockerImageFunction(
                self,
                name,
                code=lambda_.DockerImageCode.from_image_asset(
                    str(REPO),
                    file="cloud/Dockerfile",
                    cmd=[f"headvoice.cloud.handlers.{handler}"],
                ),
                architecture=lambda_.Architecture.X86_64,
                memory_size=memory,
                timeout=Duration.seconds(timeout_s),
                environment=env,
                log_group=logs.LogGroup(
                    self,
                    f"{name}Logs",
                    retention=logs.RetentionDays.TWO_WEEKS,
                    removal_policy=RemovalPolicy.DESTROY,
                ),
            )

        f_auth = fn("Authorizer", "authorizer", 256, 5)
        f_sup = fn("Supervise", "supervise", 512, 30)
        f_sug = fn("Suggest", "suggest", 512, 30)
        f_cal = fn("Calibrate", "calibrate", 2048, 120)
        f_rep = fn("Replay", "replay_handler", 2048, 300)
        f_aud = fn("AuditFn", "audit", 256, 10)
        f_jpage = fn("JudgePage", "judge_page", 256, 5)
        f_jlist = fn("JudgeDatasets", "judge_datasets", 256, 10)
        f_jrep = fn("JudgeReplay", "judge_replay", 2048, 300)

        # Least privilege ------------------------------------------------------------------
        token.grant_read(f_auth)
        bedrock_stmt = iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=[
                f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{MODEL_PROFILE}",
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
            ],
        )
        guard_stmt = iam.PolicyStatement(
            actions=["bedrock:ApplyGuardrail"], resources=[guardrail.attr_guardrail_arn]
        )
        for f in (f_sup, f_sug, f_rep, f_jrep):  # only the agent paths may call Bedrock
            f.add_to_role_policy(bedrock_stmt)
            f.add_to_role_policy(guard_stmt)
        bucket.grant_put(f_cal, "calibration-clips/*")
        bucket.grant_read(f_cal, "calibration-clips/*")
        bucket.grant_delete(f_cal, "calibration-clips/*")
        for f in (f_rep,):
            bucket.grant_read(f, "datasets/*")
        for f in (f_rep, f_jrep):
            bucket.grant_read(f, "sample/*")
            bucket.grant_read_write(f, "cache/*")  # cached agent replays
        f_jlist.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["sample/*"]}},
            )
        )
        for f in (f_sup, f_cal, f_aud):
            table.grant_write_data(f)
        table.grant_read_data(f_aud)

        # API ------------------------------------------------------------------------------
        auth = authorizers.HttpLambdaAuthorizer(
            "TokenAuth",
            f_auth,
            response_types=[authorizers.HttpLambdaResponseType.SIMPLE],
            results_cache_ttl=Duration.minutes(5),
            identity_source=["$request.header.Authorization"],
        )
        api = apigw.HttpApi(self, "Api", api_name="headvoice")
        li = integrations.HttpLambdaIntegration
        private = [
            ("/supervise", apigw.HttpMethod.POST, f_sup),
            ("/suggest", apigw.HttpMethod.POST, f_sug),
            ("/calibrate", apigw.HttpMethod.POST, f_cal),
            ("/replay", apigw.HttpMethod.POST, f_rep),
            ("/audit", apigw.HttpMethod.POST, f_aud),
            ("/audit/{session_id}", apigw.HttpMethod.GET, f_aud),
        ]
        for path, method, f in private:
            api.add_routes(
                path=path,
                methods=[method],
                integration=li(f"{f.node.id}Int{method.value}", f),
                authorizer=auth,
            )
        public = [
            ("/judge", apigw.HttpMethod.GET, f_jpage),
            ("/judge/datasets", apigw.HttpMethod.GET, f_jlist),
            ("/judge/replay", apigw.HttpMethod.POST, f_jrep),
        ]
        public_routes = []
        for path, method, f in public:
            public_routes += api.add_routes(
                path=path, methods=[method], integration=li(f"{f.node.id}Int", f)
            )
        stage = api.default_stage.node.default_child  # throttling on the default stage
        for r in public_routes:  # per-route settings need the routes to exist first
            stage.add_dependency(r.node.default_child)
        stage.default_route_settings = apigw.CfnStage.RouteSettingsProperty(
            throttling_burst_limit=20, throttling_rate_limit=10
        )
        stage.route_settings = {
            "POST /judge/replay": {"ThrottlingBurstLimit": 2, "ThrottlingRateLimit": 0.5}
        }

        # Dashboard -------------------------------------------------------------------------
        def m(name: str, stat: str = "Average", **dims) -> cw.Metric:
            return cw.Metric(
                namespace="HeadVoice",
                metric_name=name,
                statistic=stat,
                dimensions_map=dims or None,
                period=Duration.minutes(5),
            )

        dash = cw.Dashboard(self, "Dashboard", dashboard_name="HeadVoice")
        dash.add_widgets(
            cw.GraphWidget(
                title="Supervisor latency (ms)",
                left=[
                    m("AgentLatencyMs", "p50", Agent="supervisor"),
                    m("AgentLatencyMs", "p95", Agent="supervisor"),
                ],
            ),
            cw.GraphWidget(
                title="Estimated Bedrock cost (USD)",
                left=[m("EstimatedCostUSD", "Sum", Agent="supervisor")],
            ),
            cw.GraphWidget(
                title="Agent behaviour",
                left=[
                    m("AgentNoChange", "Sum", Agent="supervisor"),
                    m("AgentToolCalls", "Sum", Agent="supervisor"),
                ],
            ),
        )
        dash.add_widgets(
            cw.GraphWidget(
                title="Device-side rejections",
                left=[
                    m("ToolCallRejections", "Sum", Agent="agent"),
                    m("StaleDiscards", "Sum", Agent="agent"),
                    m("ToolCallRejections", "Sum", Agent="rule"),
                ],
            ),
            cw.GraphWidget(
                title="Calibration (ms)", left=[m("CalibrationMs", "Average", Job="calibrate")]
            ),
            cw.GraphWidget(
                title="Lambda errors",
                left=[f.metric_errors() for f in (f_sup, f_sug, f_cal, f_rep, f_aud)],
            ),
        )

        # Public website (landing, download, privacy) --------------------------------------
        site_bucket = s3.Bucket(
            self,
            "Site",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        site = cloudfront.Distribution(
            self,
            "SiteCdn",
            default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.SECURITY_HEADERS,
            ),
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            comment="HeadVoice website",
        )
        site_dir = REPO / "web" / "site"
        if site_dir.exists():  # built by scripts/build_site.py
            s3deploy.BucketDeployment(
                self,
                "SiteDeploy",
                sources=[s3deploy.Source.asset(str(site_dir))],
                destination_bucket=site_bucket,
                distribution=site,
                distribution_paths=["/*"],
            )
        CfnOutput(self, "SiteUrl", value=f"https://{site.distribution_domain_name}")

        CfnOutput(self, "JudgeReplayFunction", value=f_jrep.function_name)
        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "JudgeUrl", value=f"{api.api_endpoint}/judge")
        CfnOutput(self, "TokenSecretArn", value=token.secret_arn)
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "GuardrailId", value=guardrail.attr_guardrail_id)
