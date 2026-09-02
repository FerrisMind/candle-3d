# Full Lux3D server smoke test for external-app API surface.
param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$AssetsDir = "G:\candle-3d\test-assets",
    [int]$PollSeconds = 2,
    [int]$MaxPolls = 600
)

$ErrorActionPreference = "Stop"
$results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    $results.Add([pscustomobject]@{ Name = $Name; Ok = $Ok; Detail = $Detail })
    $mark = if ($Ok) { "PASS" } else { "FAIL" }
    Write-Host "[$mark] $Name :: $Detail"
}

function Invoke-Json {
    param([string]$Method, [string]$Path, $Body = $null, [hashtable]$Headers = @{})
    $uri = "$BaseUrl$Path"
    $params = @{
        Uri = $uri
        Method = $Method
        Headers = $Headers
    }
    if ($null -ne $Body) {
        $params.Body = ($Body | ConvertTo-Json -Depth 8)
        $params.ContentType = "application/json"
    }
    return Invoke-RestMethod @params
}

function Invoke-Multipart {
    param([string]$Path, [hashtable]$Fields)
    $args = @("-s", "-X", "POST", "$BaseUrl$Path")
    foreach ($key in $Fields.Keys) {
        $value = $Fields[$key]
        if ($value -is [System.IO.FileInfo]) {
            $args += "-F"
            $args += "$key=@$($value.FullName)"
        } else {
            $args += "-F"
            $args += "$key=$value"
        }
    }
    $raw = & curl.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for POST $Path (exit $LASTEXITCODE)"
    }
    return $raw | ConvertFrom-Json
}

function Upload-File {
    param([string]$FilePath, [string]$Purpose = "generation")
    $raw = curl.exe -s -X POST "$BaseUrl/v1/files" -F "purpose=$Purpose" -F "file=@$FilePath"
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed uploading file (exit $LASTEXITCODE)"
    }
    return $raw | ConvertFrom-Json
}

function Test-CancelQueuedJob {
    param([string]$Name)
    try {
        $queued = Invoke-Multipart "/v1/point-clouds/generations" @{
            model = "pi3"
            source = (Get-Item $framesZip)
            options = '{"interval":1}'
        }
        $cancelled = Invoke-Json POST "/v1/generations/$($queued.id)/cancel"
        $ok = $cancelled.status -in @("cancelled", "queued", "in_progress")
        Add-Result "$Name cancel queued job" $ok "status=$($cancelled.status)"
        Invoke-Json DELETE "/v1/generations/$($queued.id)" | Out-Null
    } catch {
        Add-Result "$Name cancel queued job" $false $_.Exception.Message
    }
}

function Wait-Generation {
    param([string]$Id)
    for ($i = 0; $i -lt $MaxPolls; $i++) {
        $job = Invoke-Json GET "/v1/generations/$Id"
        if ($job.status -in @("completed", "failed", "cancelled")) {
            return $job
        }
        Start-Sleep -Seconds $PollSeconds
    }
    throw "timeout waiting for generation $Id"
}

$meshImage = Join-Path $AssetsDir "mesh-input.png"
$framesZip = Join-Path $AssetsDir "pi3-frames.zip"
$frameImage = Join-Path $AssetsDir "pi3-frames\000000.png"

Write-Host "=== Lux3D server full API test ==="
Write-Host "Base URL: $BaseUrl"

try {
    $root = Invoke-Json GET "/"
    Add-Result "GET /" ($root.object -eq "lux3d.server") "version=$($root.version)"
} catch {
    Add-Result "GET /" $false $_.Exception.Message
}

try {
    $health = Invoke-WebRequest -Uri "$BaseUrl/health" -UseBasicParsing
    Add-Result "GET /health" ($health.StatusCode -eq 200) $health.Content
} catch {
    Add-Result "GET /health" $false $_.Exception.Message
}

try {
    $metrics = Invoke-WebRequest -Uri "$BaseUrl/metrics" -UseBasicParsing
    Add-Result "GET /metrics" ($metrics.StatusCode -eq 200 -and $metrics.Content -match "http_requests_total") "bytes=$($metrics.RawContentLength)"
} catch {
    Add-Result "GET /metrics" $false $_.Exception.Message
}

try {
    $info = Invoke-Json GET "/v1/system/info"
    Add-Result "GET /v1/system/info" ($info.object -eq "system.info") "device=$($info.device.backend)"
} catch {
    Add-Result "GET /v1/system/info" $false $_.Exception.Message
}

try {
    $doctor = Invoke-Json POST "/v1/system/doctor"
    Add-Result "POST /v1/system/doctor" ($doctor.object -eq "system.doctor") "ok=$($doctor.ok)"
} catch {
    Add-Result "POST /v1/system/doctor" $false $_.Exception.Message
}

try {
    $models = Invoke-Json GET "/v1/models"
    Add-Result "GET /v1/models" ($models.data.Count -ge 3) "count=$($models.data.Count)"
} catch {
    Add-Result "GET /v1/models" $false $_.Exception.Message
}

foreach ($modelId in @("pi3", "pi3x", "triposr")) {
    try {
        $status = Invoke-Json POST "/v1/models/status" @{ model_id = $modelId }
        Add-Result "POST /v1/models/status ($modelId)" ($null -ne $status.status) "status=$($status.status)"
    } catch {
        Add-Result "POST /v1/models/status ($modelId)" $false $_.Exception.Message
    }
}

# Files API
$fileId = $null
try {
    $upload = Upload-File -FilePath $meshImage
    $fileId = $upload.id
    Add-Result "POST /v1/files" ($upload.object -eq "file") "id=$fileId bytes=$($upload.bytes)"
} catch {
    Add-Result "POST /v1/files" $false $_.Exception.Message
}

if ($fileId) {
    try {
        $listed = Invoke-Json GET "/v1/files?limit=5"
        Add-Result "GET /v1/files" ($listed.data.Count -ge 1) "count=$($listed.data.Count)"
    } catch {
        Add-Result "GET /v1/files" $false $_.Exception.Message
    }

    try {
        $meta = Invoke-Json GET "/v1/files/$fileId"
        Add-Result "GET /v1/files/{id}" ($meta.id -eq $fileId) "filename=$($meta.filename)"
    } catch {
        Add-Result "GET /v1/files/{id}" $false $_.Exception.Message
    }

    try {
        $content = Invoke-WebRequest -Uri "$BaseUrl/v1/files/$fileId/content" -UseBasicParsing
        Add-Result "GET /v1/files/{id}/content" ($content.StatusCode -eq 200 -and $content.RawContentLength -gt 0) "bytes=$($content.RawContentLength)"
    } catch {
        Add-Result "GET /v1/files/{id}/content" $false $_.Exception.Message
    }
}

function Test-PointCloudModel {
    param([string]$Model, [switch]$UseJson, [switch]$UseFileId)

    $name = "point-cloud/$Model"
    try {
        if ($UseJson) {
            $body = @{
                model = $Model
                options = @{ interval = 1 }
                webhook_url = $null
            }
            if ($UseFileId) {
                $body.source_file_id = $fileId
            } else {
                $body.source_url = $null
                $body.source_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($frameImage))
                $body.source_filename = "000000.png"
            }
            $accepted = Invoke-Json POST "/v1/point-clouds/generations" $body
        } else {
            $accepted = Invoke-Multipart "/v1/point-clouds/generations" @{
                model = $Model
                source = (Get-Item $framesZip)
                options = '{"interval":1}'
            }
        }
        $jobId = $accepted.id
        Add-Result "$name create" ($accepted.status -eq "queued") "id=$jobId"

        $final = Wait-Generation $jobId
        Add-Result "$name complete" ($final.status -eq "completed") "status=$($final.status) error=$($final.error.message)"

        if ($final.status -eq "completed") {
            $asset = Invoke-WebRequest -Uri "$BaseUrl/v1/generations/$jobId/content" -UseBasicParsing
            Add-Result "$name content" ($asset.StatusCode -eq 200 -and $asset.RawContentLength -gt 100) "bytes=$($asset.RawContentLength)"
        }

        $deleted = Invoke-Json DELETE "/v1/generations/$jobId"
        Add-Result "$name delete" ($deleted.id -eq $jobId) "status=$($deleted.status)"
    } catch {
        Add-Result "$name flow" $false $_.Exception.Message
    }
}

function Test-MeshModel {
    param([switch]$UseJson, [switch]$UseFileId)

    $name = "mesh/triposr"
    try {
        if ($UseJson) {
            $body = @{
                model = "triposr"
                options = @{ mc_resolution = 64; mc_threshold = 25.0 }
            }
            if ($UseFileId) {
                $body.source_file_id = $fileId
            } else {
                $body.source_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($meshImage))
                $body.source_filename = "mesh-input.png"
            }
            $accepted = Invoke-Json POST "/v1/meshes/generations" $body
        } else {
            $accepted = Invoke-Multipart "/v1/meshes/generations" @{
                model = "triposr"
                source = (Get-Item $meshImage)
                options = '{"mc_resolution":64,"mc_threshold":25.0}'
            }
        }

        $jobId = $accepted.id
        Add-Result "$name create" ($accepted.status -eq "queued") "id=$jobId"

        $final = Wait-Generation $jobId
        Add-Result "$name complete" ($final.status -eq "completed") "status=$($final.status) error=$($final.error.message)"

        if ($final.status -eq "completed") {
            $asset = Invoke-WebRequest -Uri "$BaseUrl/v1/generations/$jobId/content" -UseBasicParsing
            Add-Result "$name content" ($asset.StatusCode -eq 200 -and $asset.RawContentLength -gt 100) "bytes=$($asset.RawContentLength)"
        }

        $deleted = Invoke-Json DELETE "/v1/generations/$jobId"
        Add-Result "$name delete" ($deleted.id -eq $jobId) "status=$($deleted.status)"
    } catch {
        Add-Result "$name flow" $false $_.Exception.Message
    }
}

Test-PointCloudModel -Model "pi3"
Test-PointCloudModel -Model "pi3x"
Test-PointCloudModel -Model "pi3" -UseJson
Test-PointCloudModel -Model "pi3x" -UseJson
if ($fileId) {
    Test-PointCloudModel -Model "pi3" -UseJson -UseFileId
}

Test-MeshModel
Test-MeshModel -UseJson
if ($fileId) {
    Test-MeshModel -UseJson -UseFileId
}

Test-CancelQueuedJob -Name "generations"

foreach ($modelId in @("pi3", "pi3x", "triposr")) {
    try {
        $reload = Invoke-Json POST "/v1/models/reload" @{ model_id = $modelId }
        Add-Result "POST /v1/models/reload ($modelId)" ($reload.status -eq "loaded") "status=$($reload.status)"
    } catch {
        Add-Result "POST /v1/models/reload ($modelId)" $false $_.Exception.Message
    }
}

if ($fileId) {
    try {
        $removed = Invoke-Json DELETE "/v1/files/$fileId"
        Add-Result "DELETE /v1/files/{id}" ($removed.id -eq $fileId) "filename=$($removed.filename)"
    } catch {
        Add-Result "DELETE /v1/files/{id}" $false $_.Exception.Message
    }
}

try {
    $gens = Invoke-Json GET "/v1/generations?limit=5"
    Add-Result "GET /v1/generations" ($gens.object -eq "list") "count=$($gens.data.Count)"
} catch {
    Add-Result "GET /v1/generations" $false $_.Exception.Message
}

$passed = ($results | Where-Object Ok).Count
$failed = ($results | Where-Object { -not $_.Ok }).Count
Write-Host ""
Write-Host "=== SUMMARY: passed=$passed failed=$failed total=$($results.Count) ==="
if ($failed -gt 0) {
    $results | Where-Object { -not $_.Ok } | Format-Table -AutoSize
    exit 1
}
exit 0
