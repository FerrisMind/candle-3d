# Open a TripoSR OBJ artifact for demo preview.
# Primary: local three.js viewer in the default browser.
# Fallback: Blender (if browser viewer cannot start, or -Blender is set).
#
# Usage:
#   .\scripts\demo_open_obj.ps1
#   .\scripts\demo_open_obj.ps1 -Path .\demo-out\triposr-vulkan.obj
#   .\scripts\demo_open_obj.ps1 -Path .\demo-out\triposr-vulkan.obj -Blender
param(
    [string]$Path = "",
    [int]$Port = 8765,
    [switch]$Blender,
    [string]$BlenderExe = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not $Path) {
    $Path = Join-Path $RepoRoot "demo-out\triposr-vulkan.obj"
}
$Path = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)

if (-not (Test-Path -LiteralPath $Path)) {
    throw "OBJ not found: $Path (run .\scripts\demo_triposr_vulkan.ps1 first)"
}
if ([IO.Path]::GetExtension($Path) -ne ".obj") {
    throw "Expected a .obj file, got: $Path"
}

function Find-Blender {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit)) {
            throw "BlenderExe not found: $Explicit"
        }
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    $cmd = Get-Command blender -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $roots = @(
        "${env:ProgramFiles}\Blender Foundation",
        "${env:ProgramFiles(x86)}\Blender Foundation",
        "$env:LOCALAPPDATA\Programs\Blender Foundation"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $hit = Get-ChildItem -LiteralPath $root -Recurse -Filter "blender.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Open-InBlender {
    param([string]$ObjPath, [string]$Exe)
    if (-not $Exe) {
        throw "Blender not found. Install Blender or pass -BlenderExe `"C:\Program Files\Blender Foundation\Blender X.Y\blender.exe`""
    }
    $py = Join-Path $PSScriptRoot "demo_open_obj_blender.py"
    if (-not (Test-Path -LiteralPath $py)) {
        throw "Blender helper missing: $py"
    }
    Write-Host "Opening in Blender (Gradio orientation + vertex colors): $Exe"
    Write-Host "  file: $ObjPath"
    # --python runs after factory startup; "--" separates script args.
    Start-Process -FilePath $Exe -ArgumentList @(
        "--python", $py,
        "--", $ObjPath
    )
}

function Test-PortFree {
    param([int]$P)
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $P)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Open-InBrowserViewer {
    param([string]$ObjPath, [int]$PreferredPort)

    $viewerSrc = Join-Path $PSScriptRoot "demo_obj_viewer.html"
    if (-not (Test-Path -LiteralPath $viewerSrc)) {
        throw "Viewer template missing: $viewerSrc"
    }

    $stage = Join-Path $env:TEMP ("lux3d-obj-preview-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $stage | Out-Null
    Copy-Item -LiteralPath $viewerSrc -Destination (Join-Path $stage "index.html")
    Copy-Item -LiteralPath $ObjPath -Destination (Join-Path $stage "model.obj")

    $port = $PreferredPort
    if (-not (Test-PortFree $port)) {
        $port = Get-Random -Minimum 8800 -Maximum 9800
        if (-not (Test-PortFree $port)) {
            throw "No free TCP port for local preview server"
        }
    }

    $url = "http://127.0.0.1:$port/index.html?obj=model.obj"
    Write-Host "=== OBJ preview (browser / three.js) ==="
    Write-Host "file: $ObjPath"
    Write-Host "url:  $url"
    Write-Host "stop: close this window or Ctrl+C"

    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "python not found (needed for local preview server)"
    }

    Start-Process $url | Out-Null
    Push-Location $stage
    try {
        & python -m http.server $port --bind 127.0.0.1
    } finally {
        Pop-Location
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "artifact: $Path ($((Get-Item -LiteralPath $Path).Length) bytes)"

if ($Blender) {
    Open-InBlender -ObjPath $Path -Exe (Find-Blender -Explicit $BlenderExe)
    return
}

try {
    Open-InBrowserViewer -ObjPath $Path -PreferredPort $Port
} catch {
    Write-Warning "Browser preview failed: $($_.Exception.Message)"
    Write-Host "Falling back to Blender..."
    Open-InBlender -ObjPath $Path -Exe (Find-Blender -Explicit $BlenderExe)
}
