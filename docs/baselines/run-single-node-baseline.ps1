# Single-node Kivi baseline: spawn server, run kivi-lab bench matrix, stop server.
#
# Self-contained by design: the server is always stopped before this script
# exits (including on bench failure), so the calling command always
# completes and never leaves a stray process holding ports or log files.
#
# Usage (from the workspace root):
#   powershell -ExecutionPolicy Bypass -File docs/baselines/run-single-node-baseline.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$serverBin = Join-Path $root "target\release\kivi-server.exe"
$labBin = Join-Path $root "target\release\kivi-lab.exe"
$outDir = $PSScriptRoot

foreach ($bin in @($serverBin, $labBin)) {
    if (-not (Test-Path -LiteralPath $bin)) {
        throw "missing binary: $bin (run: cargo build --release -p kivi-server --features redis-compat -p kivi-lab)"
    }
}

function Get-PortOwner {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $conn) { return $null }
    return $conn.OwningProcess
}

# Starts the server in the background and waits for KIVI_READY.
# Returns the server PID; the caller must stop it via Stop-ServerPid in a
# finally block so no call ever leaves a process behind.
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

# Pre-clean: a previous interrupted run may have left a server behind.
# (Only our own baseline ports are touched.)
Stop-KiviServer 9000
Stop-KiviServer 9100

$ephemeralPid = 0
try {
    # --- Ephemeral server: pure transport numbers (native + RESP). ---
    $ephemeralPid = Start-KiviServer @("--ephemeral", "--port", "9000", "--workers", "2", "--pin", "none", "--admin", "127.0.0.1:19080", "--redis-listen", "127.0.0.1:6380") (Join-Path $outDir "server-ephemeral.log") 9000
    Invoke-Bench "native-ephemeral-set-b16-1t" "--target native --server 127.0.0.1:9000 --threads 1 --ops 5000 --warmup 500 --workload set --keys uniform:256 --prefix base:ephem:set1: --value-size b16"
    Invoke-Bench "native-ephemeral-set-b16-8t" "--target native --server 127.0.0.1:9000 --threads 8 --ops 5000 --warmup 500 --workload set --keys uniform:256 --prefix base:ephem:set8: --value-size b16"
    Invoke-Bench "native-ephemeral-get-b16-1t" "--target native --server 127.0.0.1:9000 --threads 1 --ops 5000 --warmup 500 --workload get --keys uniform:256 --prefix base:ephem:set1: --value-size b16"
    Invoke-Bench "native-ephemeral-get-b16-8t" "--target native --server 127.0.0.1:9000 --threads 8 --ops 5000 --warmup 500 --workload get --keys uniform:256 --prefix base:ephem:set8: --value-size b16"
    Invoke-Bench "native-ephemeral-set-mb1-1t" "--target native --server 127.0.0.1:9000 --threads 1 --ops 20 --warmup 0 --workload set --keys uniform:4 --prefix base:ephem:mb1: --value-size mb1"
    Invoke-Bench "resp-compat-8t" "--target resp --server 127.0.0.1:6380 --threads 8 --ops 5000 --warmup 500 --workload compat --keys uniform:256 --prefix base:ephem:resp:"
} finally {
    Stop-ServerPid $ephemeralPid
}

$durablePid = 0
try {
    # --- Durable server: production-like numbers (WAL group commit). ---
    $dataDir = Join-Path $outDir "data-durable"
    Remove-Item -LiteralPath $dataDir -Recurse -Force -ErrorAction SilentlyContinue
    $durablePid = Start-KiviServer @("--data-dir", $dataDir, "--port", "9100", "--workers", "2", "--pin", "none", "--admin", "127.0.0.1:19081", "--no-checkpoint") (Join-Path $outDir "server-durable.log") 9100
    Invoke-Bench "native-durable-set-b16-1t" "--target native --server 127.0.0.1:9100 --threads 1 --ops 5000 --warmup 500 --workload set --keys uniform:256 --prefix base:dur:set1: --value-size b16"
    Invoke-Bench "native-durable-set-b16-8t" "--target native --server 127.0.0.1:9100 --threads 8 --ops 5000 --warmup 500 --workload set --keys uniform:256 --prefix base:dur:set8: --value-size b16"
    Invoke-Bench "native-durable-get-b16-8t" "--target native --server 127.0.0.1:9100 --threads 8 --ops 5000 --warmup 500 --workload get --keys uniform:256 --prefix base:dur:set8: --value-size b16"
    Invoke-Bench "native-durable-counter-1t" "--target native --server 127.0.0.1:9100 --threads 1 --ops 2000 --warmup 200 --workload counter --keys uniform:64 --prefix base:dur:ctr1:"
    Invoke-Bench "native-durable-counter-8t" "--target native --server 127.0.0.1:9100 --threads 8 --ops 2000 --warmup 200 --workload counter --keys uniform:64 --prefix base:dur:ctr8:"
} finally {
    Stop-ServerPid $durablePid
}

Write-Output "baseline matrix complete"
