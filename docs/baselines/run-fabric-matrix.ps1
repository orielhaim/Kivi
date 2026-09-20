# Fabric benchmark matrix: all-DRAM reference vs integrated fabric under
# growing pressure. Complements run-single-node-baseline.ps1 (b16 inline
# numbers); every run here uses kb1 (1 KiB) medium values that ride the
# fabric (or chunked sidecars on cluster paths).
#
# Usage (from the workspace root):
#   powershell -ExecutionPolicy Bypass -File docs/baselines/run-fabric-matrix.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$serverBin = Join-Path $root "target\release\kivi-server.exe"
$labBin = Join-Path $root "target\release\kivi-lab.exe"
$outDir = $PSScriptRoot

foreach ($bin in @($serverBin, $labBin)) {
    if (-not (Test-Path -LiteralPath $bin)) {
        throw "missing binary: $bin (run: cargo build --release -p kivi-server -p kivi-lab)"
    }
}

function Get-PortOwner {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $conn) { return $null }
    return $conn.OwningProcess
}

function Start-KiviServer {
    param([string[]]$ServerArgsArray, [string]$LogPath, [int]$Port)
    Remove-Item -LiteralPath $LogPath -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $serverBin -ArgumentList $ServerArgsArray `
        -RedirectStandardOutput $LogPath -WindowStyle Hidden -PassThru
    for ($i = 0; $i -lt 40; $i++) {
        if ($proc.HasExited) {
            throw "server on port $Port exited during startup (see $LogPath)"
        }
        if (Test-Path -LiteralPath $LogPath) {
            if (Select-String -Path $LogPath -Pattern "KIVI_READY" -Quiet) { return $proc.Id }
        }
        Start-Sleep -Milliseconds 500
    }
    throw "server on port $Port never reported KIVI_READY (see $LogPath)"
}

function Stop-ServerPid {
    param([int]$ServerPid)
    if ($ServerPid -le 0) { return }
    Stop-Process -Id $ServerPid -Force -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 20; $i++) {
        if ($null -eq (Get-Process -Id $ServerPid -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 500
    }
    throw "server process $ServerPid did not exit"
}

function Stop-KiviServer {
    param([int]$Port)
    $owner = Get-PortOwner -Port $Port
    if ($null -ne $owner) {
        Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
        for ($i = 0; $i -lt 20; $i++) {
            if ($null -eq (Get-PortOwner -Port $Port)) { return }
            Start-Sleep -Milliseconds 500
        }
        throw "server process $owner on port $Port did not exit"
    }
}

function Invoke-Bench {
    param([string]$Name, [string]$BenchSpec)
    $out = Join-Path $outDir "$Name.json"
    Remove-Item -LiteralPath $out -ErrorAction SilentlyContinue
    & $labBin bench ($BenchSpec -split " ") --format json --out $out
    if ($LASTEXITCODE -ne 0) { throw "bench $Name failed (exit $LASTEXITCODE)" }
    Write-Output "wrote $out"
}

Stop-KiviServer 9200
Stop-KiviServer 9210

$ephemeralPid = 0
try {
    # --- Reference: b16 inline (all-DRAM, no fabric involvement). ---
    $ephemeralPid = Start-KiviServer @("--ephemeral", "--port", "9200", "--workers", "2", "--pin", "none", "--admin", "127.0.0.1:19090") (Join-Path $outDir "server-fabric-ref.log") 9200
    Invoke-Bench "fabric-ref-set-b16-8t" "--target native --server 127.0.0.1:9200 --threads 8 --ops 5000 --warmup 500 --workload set --keys uniform:256 --prefix base:fab:ref8: --value-size b16"
    Invoke-Bench "fabric-ref-get-b16-8t" "--target native --server 127.0.0.1:9200 --threads 8 --ops 5000 --warmup 500 --workload get --keys uniform:256 --prefix base:fab:ref8: --value-size b16"
    # --- Fabric, roomy arena: medium values, no pressure. ---
    Invoke-Bench "fabric-roomy-set-kb1-8t" "--target native --server 127.0.0.1:9200 --threads 8 --ops 5000 --warmup 500 --workload set --keys uniform:256 --prefix base:fab:room8: --value-size kb1"
    Invoke-Bench "fabric-roomy-get-kb1-8t" "--target native --server 127.0.0.1:9200 --threads 8 --ops 5000 --warmup 500 --workload get --keys uniform:256 --prefix base:fab:room8: --value-size kb1"
} finally {
    Stop-ServerPid $ephemeralPid
}

$pressuredPid = 0
try {
    # --- Fabric, 1 MiB arena over 1.5 MiB of medium keys: the drain
    # mostly keeps up, so sets succeed with visible movement underneath
    # (2x overload sheds by design; see the report). Single-thread
    # seeding is the gentlest burst.
    $pressuredPid = Start-KiviServer @("--ephemeral", "--port", "9210", "--workers", "2", "--pin", "none", "--admin", "127.0.0.1:19091", "--fabric-arena-bytes", "1048576") (Join-Path $outDir "server-fabric-pressure.log") 9210
    Invoke-Bench "fabric-pressure-set-kb1-1t" "--target native --server 127.0.0.1:9210 --threads 1 --ops 1500 --warmup 0 --workload set --keys uniform:1536 --prefix base:fab:pres1: --value-size kb1"
    Invoke-Bench "fabric-pressure-get-kb1-1t" "--target native --server 127.0.0.1:9210 --threads 1 --ops 1500 --warmup 0 --workload get --keys uniform:1536 --prefix base:fab:pres1: --value-size kb1"
} finally {
    Stop-ServerPid $pressuredPid
}

$durablePid = 0
try {
    # --- Durable (file NVMe demotion device): roomy and constrained. ---
    $dataDir = Join-Path $outDir "data-fabric"
    Remove-Item -LiteralPath $dataDir -Recurse -Force -ErrorAction SilentlyContinue
    $durablePid = Start-KiviServer @("--data-dir", $dataDir, "--port", "9200", "--workers", "2", "--pin", "none", "--admin", "127.0.0.1:19090", "--no-checkpoint") (Join-Path $outDir "server-fabric-durable.log") 9200
    Invoke-Bench "fabric-durable-set-kb1-8t" "--target native --server 127.0.0.1:9200 --threads 8 --ops 5000 --warmup 500 --workload set --keys uniform:256 --prefix base:fab:dur8: --value-size kb1"
    Invoke-Bench "fabric-durable-get-kb1-8t" "--target native --server 127.0.0.1:9200 --threads 8 --ops 5000 --warmup 500 --workload get --keys uniform:256 --prefix base:fab:dur8: --value-size kb1"
} finally {
    Stop-ServerPid $durablePid
}

Write-Output "fabric matrix complete"
