# Supplementary curl-based tests for Windows PowerShell 5.1
$Base = "http://127.0.0.1:8080"
$Assets = "G:\candle-3d\test-assets"
$log = New-Object System.Collections.Generic.List[string]

function Log([string]$name, [bool]$ok, [string]$detail) {
    $mark = if ($ok) { "PASS" } else { "FAIL" }
    $line = "[$mark] $name :: $detail"
    $log.Add($line)
    Write-Host $line
}

function Get-Json([string]$path) {
    return (curl.exe -s "$Base$path") | ConvertFrom-Json
}

function Post-Json([string]$path, [string]$body) {
    return (curl.exe -s -X POST "$Base$path" -H "Content-Type: application/json" -d $body) | ConvertFrom-Json
}

Write-Host "=== curl supplementary tests ==="

# Files API
$fileResp = curl.exe -s -X POST "$Base/v1/files" -F "purpose=generation" -F "file=@$Assets/mesh-input.png"
$file = $fileResp | ConvertFrom-Json
Log "POST /v1/files (curl)" ($file.object -eq "file") "id=$($file.id)"

$list = Get-Json "/v1/files?limit=5"
Log "GET /v1/files" ($list.data.Count -ge 1) "count=$($list.data.Count)"

$meta = Get-Json "/v1/files/$($file.id)"
Log "GET /v1/files/{id}" ($meta.id -eq $file.id) "filename=$($meta.filename)"

$content = curl.exe -s -o NUL -w "%{http_code} %{size_download}" "$Base/v1/files/$($file.id)/content"
Log "GET /v1/files/{id}/content" ($content -match "^200 ") $content

# multipart generations (single-quoted JSON for curl on Windows)
$pi3Resp = curl.exe -s -X POST "$Base/v1/point-clouds/generations" -F "model=pi3" -F "source=@$Assets/pi3-frames.zip" -F 'options={"interval":1}'
$pi3 = $pi3Resp | ConvertFrom-Json
Log "POST /v1/point-clouds/generations pi3 multipart" ($pi3.status -eq "queued") "id=$($pi3.id)"

$pi3xResp = curl.exe -s -X POST "$Base/v1/point-clouds/generations" -F "model=pi3x" -F "source=@$Assets/pi3-frames.zip" -F 'options={"interval":1}'
$pi3x = $pi3xResp | ConvertFrom-Json
Log "POST /v1/point-clouds/generations pi3x multipart" ($pi3x.status -eq "queued") "id=$($pi3x.id)"

$meshResp = curl.exe -s -X POST "$Base/v1/meshes/generations" -F "model=triposr" -F "source=@$Assets/mesh-input.png" -F 'options={"mc_resolution":64,"mc_threshold":25.0}'
$mesh = $meshResp | ConvertFrom-Json
Log "POST /v1/meshes/generations triposr multipart" ($mesh.status -eq "queued") "id=$($mesh.id)"

# cancel in-progress job quickly
$cancelJobResp = curl.exe -s -X POST "$Base/v1/point-clouds/generations" -F "model=pi3" -F "source=@$Assets/pi3-frames.zip" -F 'options={"interval":1}'
$cancelJob = $cancelJobResp | ConvertFrom-Json
$cancelResp = curl.exe -s -w "`nHTTP:%{http_code}" -X POST "$Base/v1/generations/$($cancelJob.id)/cancel"
$cancelOk = ($cancelResp -like '*cancelled*') -or ($cancelResp -like '*queued*') -or ($cancelResp -like '*in_progress*')
Log "POST /v1/generations/{id}/cancel (queued/in-progress)" $cancelOk $cancelResp

function Wait-Job([string]$id) {
    for ($i = 0; $i -lt 300; $i++) {
        $job = Get-Json "/v1/generations/$id"
        if ($job.status -in @("completed", "failed", "cancelled")) { return $job }
        Start-Sleep -Seconds 2
    }
    throw "timeout $id"
}

foreach ($pair in @(@("pi3 multipart", $pi3.id), @("pi3x multipart", $pi3x.id), @("triposr multipart", $mesh.id))) {
    $name = $pair[0]; $id = $pair[1]
    if (-not $id) {
        Log "$name complete" $false "job was not queued"
        continue
    }
    $final = Wait-Job $id
    Log "$name complete" ($final.status -eq "completed") "status=$($final.status)"
    if ($final.status -eq "completed") {
        $dl = curl.exe -s -o NUL -w "%{http_code} %{size_download}" "$Base/v1/generations/$id/content"
        Log "$name content" ($dl -match "^200 ") $dl
    }
    $del = curl.exe -s -X DELETE "$Base/v1/generations/$id"
    $deleted = $del | ConvertFrom-Json
    Log "$name delete" ($deleted.id -eq $id) "status=$($deleted.status)"
}

# JSON with source_file_id
$pcJson = Post-Json "/v1/point-clouds/generations" "{`"model`":`"pi3`",`"source_file_id`":`"$($file.id)`",`"options`":{`"interval`":1}}"
Log "POST /v1/point-clouds/generations source_file_id" ($pcJson.status -eq "queued") "id=$($pcJson.id)"
$pcFinal = Wait-Job $pcJson.id
Log "source_file_id pi3 complete" ($pcFinal.status -eq "completed") "status=$($pcFinal.status)"
curl.exe -s -X DELETE "$Base/v1/generations/$($pcJson.id)" | Out-Null

$meshJson = Post-Json "/v1/meshes/generations" "{`"model`":`"triposr`",`"source_file_id`":`"$($file.id)`",`"options`":{`"mc_resolution`":64,`"mc_threshold`":25.0}}"
Log "POST /v1/meshes/generations source_file_id" ($meshJson.status -eq "queued") "id=$($meshJson.id)"
if ($meshJson.id) {
    $meshFinal = Wait-Job $meshJson.id
    # source_file_id triposr (workspace + staged source fix)
    Log "source_file_id triposr complete" ($meshFinal.status -eq "completed") "status=$($meshFinal.status)"
    curl.exe -s -X DELETE "$Base/v1/generations/$($meshJson.id)" | Out-Null
}

# model unload/status after reload
foreach ($modelId in @("pi3", "pi3x", "triposr")) {
    $unload = Post-Json "/v1/models/unload" "{`"model_id`":`"$modelId`"}"
    Log "POST /v1/models/unload ($modelId)" ($unload.status -eq "unloaded") "status=$($unload.status)"
}

$removed = curl.exe -s -X DELETE "$Base/v1/files/$($file.id)"
$removedObj = $removed | ConvertFrom-Json
Log "DELETE /v1/files/{id}" ($removedObj.id -eq $file.id) "filename=$($removedObj.filename)"

$passed = ($log | Where-Object { $_ -match '\[PASS\]' }).Count
$failed = ($log | Where-Object { $_ -match '\[FAIL\]' }).Count
Write-Host ""
Write-Host "=== CURL SUMMARY: passed=$passed failed=$failed total=$($log.Count) ==="
if ($failed -gt 0) { exit 1 }
