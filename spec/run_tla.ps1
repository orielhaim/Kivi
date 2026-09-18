#Requires -Version 7
<#
.SYNOPSIS
  TLC check runner for the roster_lease formal model (fast gate only).
.DESCRIPTION
  Runs each TLC configuration in an isolated scratch directory (never in
  spec/, so TLC states/traces never pollute the repo) and verifies the
  verdict against the declared expectation:
    - the canonical model must PASS exhaustive and clean;
    - each exhibiting broken variant must FAIL with a NoStaleRead
      violation — a broken config that passes, violates a different
      invariant, or needs more than the timeout to fire is a modeling
      problem and fails the run.
  Every check here finishes in ~2 minutes or less on a development
  machine. Larger explorations (full-composition base model, deep
  variant scopes) are manual research, not gate checks: reconstruct
  them from the bounds noted in roster_lease.tla and run by hand.
.EXAMPLE
  ./spec/run_tla.ps1
  ./spec/run_tla.ps1 -Only roster_lease_broken_takeover -TimeoutSec 900
#>
param(
  [string[]]$Only = @(),
  [int]$TimeoutSec = 600,
  [string]$WorkRoot = "D:\tlc-work\runs",
  [int]$Workers = 16,
  [string]$Xmx = "24g",
  [switch]$KeepWorkdir
)

$ErrorActionPreference = "Stop"
$SpecDir = $PSScriptRoot
$Jar = Join-Path $SpecDir "..\.research\tla2tools.jar" | Resolve-Path | ForEach-Object { $_.Path }

# Fast gate set: canonical exhaustive plus the two exhibiting
# negatives. Everything here verdicts in ~2 minutes or less.
if ($Only.Count -eq 0) {
  $Only = @(
    "roster_lease_canonical",
    "roster_lease_broken_floor",
    "roster_lease_broken_takeover"
  )
}

# check name -> expected verdict: "pass" or the invariant that must be violated.
$Expectations = @{
  "roster_lease_canonical"         = "pass"
  "roster_lease_broken_floor"      = "NoStaleRead"
  "roster_lease_broken_takeover"   = "NoStaleRead"
}

$Failures = 0
$Inconclusive = 0
$Rows = @()

foreach ($name in $Only) {
  if (-not $Expectations.ContainsKey($name)) { throw "unknown check: $name" }
  $expect = $Expectations[$name]
  $work = Join-Path $WorkRoot $name
  if (Test-Path $work) { Remove-Item $work -Recurse -Force }
  New-Item -ItemType Directory -Path $work | Out-Null
  Copy-Item (Join-Path $SpecDir "roster_lease.tla") $work
  Copy-Item (Join-Path $SpecDir "$name.cfg") $work
  $meta = Join-Path $work "meta"
  $outFile = Join-Path $work "tlc-out.txt"
  $errFile = Join-Path $work "tlc-err.txt"

  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $proc = Start-Process java -ArgumentList @(
    "-Xmx$Xmx", "-XX:+UseParallelGC",
    "-cp", $Jar, "tlc2.TLC",
    "-config", "$name.cfg", "-workers", "$Workers",
    "-metadir", $meta, "roster_lease"
  ) -WorkingDirectory $work -NoNewWindow -PassThru `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile
  $finished = $proc.WaitForExit($TimeoutSec * 1000)
  if (-not $finished) { Stop-Process -Id $proc.Id -Force }
  $sw.Stop()

  $text = ""
  if (Test-Path $outFile) { $text += Get-Content $outFile -Raw }
  if (Test-Path $errFile) { $text += "`n" + (Get-Content $errFile -Raw) }

  $violated = $null
  if ($text -match "Invariant (\w+) is violated") { $violated = $Matches[1] }
  $stateMatches = [regex]::Matches($text, "([\d,]+) distinct states found")
  $states = if ($stateMatches.Count -gt 0) { $stateMatches[$stateMatches.Count - 1].Groups[1].Value } else { "?" }
  $depth = ([regex]::Matches($text, "(?m)^State \d+:")).Count
  $clean = $text -match "Model checking completed\. No errors? found" -or $text -match "No error has been found"
  $elapsed = [math]::Round($sw.Elapsed.TotalSeconds)

  $verdict = ""
  if (-not $finished -and $null -eq $violated) {
    if ($expect -eq "pass") { $verdict = "INCONCLUSIVE(timeout)"; $Inconclusive++ }
    else { $verdict = "FAIL(timeout without violation)"; $Failures++ }
  } elseif ($null -ne $violated -and $violated -eq $expect) {
    $verdict = "PASS(violated $violated as expected)"
  } elseif ($null -ne $violated -and $expect -eq "pass") {
    $verdict = "FAIL(unexpected $violated violation)"; $Failures++
  } elseif ($null -ne $violated) {
    $verdict = "FAIL(wrong invariant: $violated, want $expect)"; $Failures++
  } elseif ($clean -and $expect -eq "pass") {
    $verdict = "PASS(clean)"
  } elseif ($clean) {
    $verdict = "FAIL(finished clean, want $expect violation)"; $Failures++
  } else {
    $verdict = "FAIL(unparsable TLC output)"; $Failures++
  }

  $Rows += [pscustomobject]@{
    Check     = $name
    Expect    = $expect
    Verdict   = $verdict
    Invariant = ($violated ?? "-")
    Depth     = $(if ($depth -gt 0) { $depth } else { "-" })
    States    = $states
    Seconds   = $elapsed
  }

  if (-not $KeepWorkdir) { Remove-Item $work -Recurse -Force }
}

$Rows | Format-Table -AutoSize | Out-String | Write-Output
if ($Failures -gt 0) { exit 1 }
if ($Inconclusive -gt 0) { exit 2 }
exit 0
