[CmdletBinding()]
param(
    [ValidateSet('all', 'format', 'check', 'test', 'smoke')]
    [string]$Mode = 'all',
    [ValidateRange(1, 600)]
    [int]$SmokeFrames = 6,
    [string]$Asset,
    [switch]$Release
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$targetDir = Join-Path $repoRoot '.codex-target'
Set-Location $repoRoot

function Invoke-CargoStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "==> $Label" -ForegroundColor Cyan
    & cargo @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "${Label} failed with exit code $LASTEXITCODE"
    }
}

function Should-Run {
    param([string]$Step)
    return $Mode -eq 'all' -or $Mode -eq $Step
}

try {
    if (Should-Run 'format') {
        Invoke-CargoStep 'rustfmt' @('fmt', '--all', '--', '--check')
    }

    if (Should-Run 'check') {
        $checkArgs = @('check', '--target-dir', $targetDir, '--workspace')
        if ($Release) {
            $checkArgs += '--release'
        }
        Invoke-CargoStep 'workspace check' $checkArgs
    }

    if (Should-Run 'test') {
        $testArgs = @('test', '--target-dir', $targetDir, '-p', 'gr-render')
        if ($Release) {
            $testArgs += '--release'
        }
        Invoke-CargoStep 'gr-render tests' $testArgs
    }

    if (Should-Run 'smoke') {
        $oldFrames = [Environment]::GetEnvironmentVariable('REBUILD1_WINDOW_SMOKE_FRAMES', 'Process')
        $oldLog = [Environment]::GetEnvironmentVariable('RUST_LOG', 'Process')
        $oldAsset = [Environment]::GetEnvironmentVariable('REBUILD1_WINDOW_ASSET', 'Process')
        try {
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_SMOKE_FRAMES', $SmokeFrames.ToString(), 'Process')
            # Keep the bundled check readable and deterministic even when the caller has a verbose
            # RUST_LOG configured for interactive debugging.
            [Environment]::SetEnvironmentVariable('RUST_LOG', 'rebuild1=info,winit=warn', 'Process')
            if (-not [string]::IsNullOrWhiteSpace($Asset)) {
                [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_ASSET', $Asset, 'Process')
            }
            $runArgs = @('run', '--target-dir', $targetDir)
            if ($Release) {
                $runArgs += '--release'
            }
            $runArgs += @('--', '--window-smoke')
            Invoke-CargoStep "Vulkan window smoke ($SmokeFrames loaded frames)" $runArgs
        }
        finally {
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_SMOKE_FRAMES', $oldFrames, 'Process')
            [Environment]::SetEnvironmentVariable('RUST_LOG', $oldLog, 'Process')
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_ASSET', $oldAsset, 'Process')
        }
    }

    Write-Host 'Verification completed.' -ForegroundColor Green
}
catch {
    Write-Error $_
    exit 1
}
