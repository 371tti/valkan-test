[CmdletBinding()]
param(
    [ValidateSet('all', 'format', 'check', 'test', 'smoke')]
    [string]$Mode = 'all',
    [ValidateRange(1, 600)]
    [int]$SmokeFrames = 6,
    [ValidateRange(5, 600)]
    [int]$TimeoutSeconds = 120,
    [ValidateSet('performance', 'interactive', 'balanced', 'high')]
    [string]$Quality = 'balanced',
    [string]$QualitySequence,
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

function Invoke-SmokeStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds
    )

    $sequenceLabel = if ([string]::IsNullOrWhiteSpace($QualitySequence)) { 'none' } else { $QualitySequence }
    Write-Host "==> Vulkan window smoke ($SmokeFrames loaded frames, quality=$Quality, sequence=$sequenceLabel)" -ForegroundColor Cyan
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = 'cargo'
    $startInfo.WorkingDirectory = $repoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        [void]$startInfo.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw 'failed to start cargo smoke process'
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try {
                $process.Kill($true)
            }
            finally {
                throw "Vulkan window smoke timed out after $TimeoutSeconds seconds"
            }
        }

        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Host $stdout.TrimEnd()
        }
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            Write-Host $stderr.TrimEnd()
        }
        if ($process.ExitCode -ne 0) {
            throw "Vulkan window smoke failed with exit code $($process.ExitCode)"
        }

        $combined = "$stdout`n$stderr"
        if ($combined -match '(?i)Vulkan debug message|validation layer is unavailable|renderer skipped a submitted window frame') {
            throw 'Vulkan window smoke reported a validation or skipped-frame warning'
        }
        if ($combined -notmatch '(?i)Vulkan validation layer enabled') {
            throw 'Vulkan window smoke did not enable the validation layer'
        }
    }
    finally {
        $process.Dispose()
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
        $testArgs = @('test', '--target-dir', $targetDir, '--workspace')
        if ($Release) {
            $testArgs += '--release'
        }
        Invoke-CargoStep 'workspace tests' $testArgs
    }

    if (Should-Run 'smoke') {
        if ($Release) {
            throw 'Vulkan window smoke requires a debug build so validation can be enforced'
        }
        $oldFrames = [Environment]::GetEnvironmentVariable('REBUILD1_WINDOW_SMOKE_FRAMES', 'Process')
        $oldLog = [Environment]::GetEnvironmentVariable('RUST_LOG', 'Process')
        $oldAsset = [Environment]::GetEnvironmentVariable('REBUILD1_WINDOW_ASSET', 'Process')
        $oldQuality = [Environment]::GetEnvironmentVariable('REBUILD1_WINDOW_QUALITY', 'Process')
        $oldQualitySequence = [Environment]::GetEnvironmentVariable('REBUILD1_WINDOW_QUALITY_SEQUENCE', 'Process')
        try {
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_SMOKE_FRAMES', $SmokeFrames.ToString(), 'Process')
            # Validation and renderer logs are part of the smoke contract. The caller's log filter
            # must not hide them.
            [Environment]::SetEnvironmentVariable('RUST_LOG', 'info,winit=warn', 'Process')
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_QUALITY', $Quality, 'Process')
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_QUALITY_SEQUENCE', $QualitySequence, 'Process')

            $requestedAsset = $Asset
            if ([string]::IsNullOrWhiteSpace($requestedAsset)) {
                $requestedAsset = $oldAsset
            }
            if ([string]::IsNullOrWhiteSpace($requestedAsset)) {
                $requestedAsset = Join-Path $repoRoot 'assets/model.glb'
            }
            if (-not (Test-Path -LiteralPath $requestedAsset -PathType Leaf)) {
                throw "smoke asset does not exist: $requestedAsset"
            }
            $resolvedAsset = (Resolve-Path -LiteralPath $requestedAsset).Path
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_ASSET', $resolvedAsset, 'Process')

            $runArgs = @('run', '--target-dir', $targetDir)
            $runArgs += @('--', '--window-smoke')
            Invoke-SmokeStep $runArgs $TimeoutSeconds
        }
        finally {
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_SMOKE_FRAMES', $oldFrames, 'Process')
            [Environment]::SetEnvironmentVariable('RUST_LOG', $oldLog, 'Process')
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_ASSET', $oldAsset, 'Process')
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_QUALITY', $oldQuality, 'Process')
            [Environment]::SetEnvironmentVariable('REBUILD1_WINDOW_QUALITY_SEQUENCE', $oldQualitySequence, 'Process')
        }
    }

    Write-Host 'Verification completed.' -ForegroundColor Green
}
catch {
    Write-Error $_
    exit 1
}
