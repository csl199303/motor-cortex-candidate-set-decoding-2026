param(
    [string]$Python = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = "python" }
if (-not $Output) { $Output = Join-Path $root "results/reproduced" }
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)

function Invoke-Step {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $Arguments" }
}

$env:MPLBACKEND = "Agg"
Invoke-Step (Join-Path $PSScriptRoot "run_iscience_decoding.py") --output $Output
Invoke-Step (Join-Path $PSScriptRoot "run_iscience_geometry.py") --output $Output
Invoke-Step (Join-Path $PSScriptRoot "run_gnode_encoding_controls.py") --output-dir (Join-Path $Output "transferred_geometry_controls")
Invoke-Step (Join-Path $PSScriptRoot "check_iscience_analysis.py") --output $Output
Invoke-Step (Join-Path $PSScriptRoot "plot_iscience_revision.py") --results $Output
