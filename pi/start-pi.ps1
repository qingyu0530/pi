param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($env:OPENCODE_API_KEY)) {
    $userApiKey = [Environment]::GetEnvironmentVariable("OPENCODE_API_KEY", "User")
    if (-not [string]::IsNullOrWhiteSpace($userApiKey)) {
        $env:OPENCODE_API_KEY = $userApiKey
    }
}

if ([string]::IsNullOrWhiteSpace($env:OPENCODE_API_KEY) -and -not [string]::IsNullOrWhiteSpace($env:HY3_API_KEY)) {
    $env:OPENCODE_API_KEY = $env:HY3_API_KEY
}

if ([string]::IsNullOrWhiteSpace($env:OPENCODE_API_KEY)) {
    $enteredApiKey = Read-Host "请输入 OpenCode API Key"
    if ([string]::IsNullOrWhiteSpace($enteredApiKey)) {
        Write-Error "API key cannot be empty."
    }
    $env:OPENCODE_API_KEY = $enteredApiKey
    try {
        [Environment]::SetEnvironmentVariable("OPENCODE_API_KEY", $enteredApiKey, "User")
        Write-Host "API key saved for future pi launches."
    } catch {
        Write-Warning "Could not save the API key permanently; it will work for this launch only."
    }
}

$env:PI_CODING_AGENT_DIR = Join-Path $repoRoot ".pi\agent"

& node (Join-Path $repoRoot "packages\coding-agent\dist\bundle\cli.js") `
    --provider opencode-go `
    --model hy3 `
    --thinking max `
    @Arguments

exit $LASTEXITCODE
