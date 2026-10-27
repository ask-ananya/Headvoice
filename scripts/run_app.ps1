# Start the HeadVoice app connected to the deployed AWS stack (reads cdk-outputs.json).
#   .\scripts\run_app.ps1                                  # agent policy, camera 1, condition normal
#   .\scripts\run_app.ps1 -Policy rule -Condition small_moves
param(
    [ValidateSet("agent", "rule", "none")][string]$Policy = "agent",
    [int]$Camera = 1,
    [ValidateSet("normal", "adverse_light", "glare", "off_angle", "small_moves")][string]$Condition = "normal",
    [int]$Port = 8765
)
$root = Split-Path -Parent $PSScriptRoot
$out = (Get-Content (Join-Path $root "cdk-outputs.json") | ConvertFrom-Json).HeadVoice
$env:HEADVOICE_API_URL = $out.ApiUrl
$env:HEADVOICE_API_TOKEN = aws secretsmanager get-secret-value --secret-id $out.TokenSecretArn --query SecretString --output text
$env:HEADVOICE_POLICY = $Policy
$env:HEADVOICE_CAMERA = "$Camera"
$env:HEADVOICE_CONDITION = $Condition
Write-Host "HeadVoice: policy $Policy, camera $Camera, condition $Condition, http://localhost:$Port"
& (Join-Path $root ".venv\Scripts\python.exe") -m uvicorn headvoice.app.server:app --port $Port
