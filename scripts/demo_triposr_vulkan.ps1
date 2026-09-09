# Demo: TripoSR inference on Vulkan (for issue screen recordings).
# Usage:
#   .\scripts\demo_triposr_vulkan.ps1
#   .\scripts\demo_triposr_vulkan.ps1 -Source .\test-assets\mesh-input.png -Output .\demo-out\triposr-vulkan.obj
#   .\scripts\demo_triposr_vulkan.ps1 -SkipBuild   # use existing target\release\lux3d-cli.exe
param(
    [string]$Source = "",
    [string]$Output = "",
    [string]$ModelPath = "",
    [uint32]$McResolution = 256,
    [double]$McThreshold = 25.0,
    [switch]$SkipBuild,
    [switch]$OpenAfter
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not $Source) {
    $Source = Join-Path $RepoRoot "test-assets\mesh-input.png"
}
if (-not $Output) {
    $Output = Join-Path $RepoRoot "demo-out\triposr-vulkan.obj"
}
if (-not $ModelPath) {
    $ModelPath = Join-Path $RepoRoot "models\triposr"
}

$Source = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Source)
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$ModelPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ModelPath)

Write-Host "=== TripoSR Vulkan demo ==="
Write-Host "repo:    $RepoRoot"
$gpuLine = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuLine = (& nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>$null |
        Select-Object -First 1)
}
if ($gpuLine) {
    Write-Host "hostgpu: $gpuLine"
}
Write-Host "device:  vulkan (lux3d-cli prints full [device] gpu=... after init)"
Write-Host "source:  $Source"
Write-Host "model:   $ModelPath"
Write-Host "output:  $Output"
Write-Host "mc:      resolution=$McResolution threshold=$McThreshold"

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Source image not found: $Source"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model package not found: $ModelPath (expected canonical TripoSR weights)"
}

$outDir = Split-Path -Parent $Output
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
    Write-Host "created: $outDir"
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()

if ($SkipBuild) {
    $bin = Join-Path $RepoRoot "target\release\lux3d-cli.exe"
    if (-not (Test-Path -LiteralPath $bin)) {
        throw "SkipBuild set but binary missing: $bin (run without -SkipBuild once)"
    }
    Write-Host "`n[1/2] run (prebuilt) $bin"
    & $bin run triposr `
        --device vulkan `
        --source $Source `
        --model-path $ModelPath `
        --mc-resolution $McResolution `
        --mc-threshold $McThreshold `
        --output $Output
} else {
    Write-Host "`n[1/2] cargo run --release -p lux3d-cli --features vulkan"
    cargo run --release -p lux3d-cli --features vulkan -- `
        run triposr `
        --device vulkan `
        --source $Source `
        --model-path $ModelPath `
        --mc-resolution $McResolution `
        --mc-threshold $McThreshold `
        --output $Output
}

if ($LASTEXITCODE -ne 0) {
    throw "TripoSR Vulkan inference failed (exit $LASTEXITCODE)"
}

$sw.Stop()
if (-not (Test-Path -LiteralPath $Output)) {
    throw "Expected output missing: $Output"
}

$bytes = (Get-Item -LiteralPath $Output).Length
Write-Host "`n[2/2] done"
Write-Host "artifact: $Output ($bytes bytes)"
Write-Host "wall:     $([math]::Round($sw.Elapsed.TotalSeconds, 2))s"
Write-Host "preview:  .\scripts\demo_open_obj.ps1 -Path `"$Output`""

if ($OpenAfter) {
    & (Join-Path $PSScriptRoot "demo_open_obj.ps1") -Path $Output
}
