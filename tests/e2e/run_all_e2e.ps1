# =========================================================================
# HyperStream Master E2E Test Suite Runner
# Document: TEST_INFRA.md
# Standard: Zero-Emoji, Production Systems Engineering
# =========================================================================

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("1", "2", "3", "4", "all", "ALL")]
    [string]$Tier = "all",

    [Parameter(Mandatory = $false)]
    [string]$Filter = "",

    [Parameter(Mandatory = $false)]
    [switch]$VerboseOutput = $false
)

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."
$IncludeDir = Join-Path $ProjectRoot "include"
$SrcDir = Join-Path $ProjectRoot "src"
$BinDir = Join-Path $ScriptDir "bin"

if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
}

Write-Host "========================================================================="
Write-Host "  HyperStream Bare-Metal Vulkan LLM Inference Engine"
Write-Host "  Authoritative 4-Tier End-to-End (E2E) Test Suite Runner"
Write-Host "========================================================================="
Write-Host "[CONFIG] Project Root: $ProjectRoot"
Write-Host "[CONFIG] Target Tier : $Tier"
if ($Filter -ne "") {
    Write-Host "[CONFIG] Name Filter : $Filter"
}
Write-Host "-------------------------------------------------------------------------"

# Define all test suites
$AllSuites = @(
    # Tier 1: Feature Coverage (41 tests)
    @{ Tier = "1"; Name = "tier1_r1_gguf_tokenizer";   Source = "tier1_feature_coverage\test_tier1_r1_gguf_tokenizer.cpp";   Libs = @() },
    @{ Tier = "1"; Name = "tier1_r2_dynamic_topology"; Source = "tier1_feature_coverage\test_tier1_r2_dynamic_topology.cpp"; Libs = @() },
    @{ Tier = "1"; Name = "tier1_r3_mega_kernels";     Source = "tier1_feature_coverage\test_tier1_r3_mega_kernels.cpp";     Libs = @() },
    @{ Tier = "1"; Name = "tier1_r4_flash_attention";  Source = "tier1_feature_coverage\test_tier1_r4_flash_attention.cpp";  Libs = @() },
    @{ Tier = "1"; Name = "tier1_r5_spec_tree";        Source = "tier1_feature_coverage\test_tier1_r5_spec_tree.cpp";        Libs = @() },
    @{ Tier = "1"; Name = "tier1_r6_paged_kv";         Source = "tier1_feature_coverage\test_tier1_r6_paged_kv.cpp";         Libs = @() },
    @{ Tier = "1"; Name = "tier1_r7_platform_build";   Source = "tier1_feature_coverage\test_tier1_r7_platform_build.cpp";   Libs = @("-lws2_32") },

    # Tier 2: Boundary & Corner Cases (44 tests)
    @{ Tier = "2"; Name = "tier2_r1_gguf_boundary";     Source = "tier2_boundary_corner\test_tier2_r1_gguf_boundary.cpp";     Libs = @() },
    @{ Tier = "2"; Name = "tier2_r2_topology_boundary"; Source = "tier2_boundary_corner\test_tier2_r2_topology_boundary.cpp"; Libs = @() },
    @{ Tier = "2"; Name = "tier2_r3_mega_boundary";     Source = "tier2_boundary_corner\test_tier2_r3_mega_boundary.cpp";     Libs = @() },
    @{ Tier = "2"; Name = "tier2_r4_flash_boundary";    Source = "tier2_boundary_corner\test_tier2_r4_flash_boundary.cpp";    Libs = @() },
    @{ Tier = "2"; Name = "tier2_r5_spec_boundary";     Source = "tier2_boundary_corner\test_tier2_r5_spec_boundary.cpp";     Libs = @() },
    @{ Tier = "2"; Name = "tier2_r6_paged_boundary";    Source = "tier2_boundary_corner\test_tier2_r6_paged_boundary.cpp";    Libs = @() },
    @{ Tier = "2"; Name = "tier2_r7_platform_boundary"; Source = "tier2_boundary_corner\test_tier2_r7_platform_boundary.cpp"; Libs = @("-lws2_32") },

    # Tier 3: Cross-Feature Combinations (10 tests)
    @{ Tier = "3"; Name = "tier3_cross_feature";        Source = "tier3_cross_feature\test_tier3_cross_feature.cpp";        Libs = @("-lws2_32") },

    # Tier 4: Real-World Application Scenarios (8 tests)
    @{ Tier = "4"; Name = "tier4_real_world";           Source = "tier4_real_world\test_tier4_real_world.cpp";           Libs = @("-lws2_32") }
)

# Filter test suites by Tier
$ActiveSuites = @()
$SuiteMatchFilter = $false

if ($Filter -ne "") {
    foreach ($s in $AllSuites) {
        if (($Tier -eq "all" -or $Tier -eq "ALL" -or $s.Tier -eq $Tier) -and ($s.Name -like "*$Filter*" -or $s.Source -like "*$Filter*")) {
            $ActiveSuites += $s
            $SuiteMatchFilter = $true
        }
    }
}

if (-not $SuiteMatchFilter) {
    foreach ($s in $AllSuites) {
        if ($Tier -eq "all" -or $Tier -eq "ALL" -or $s.Tier -eq $Tier) {
            $ActiveSuites += $s
        }
    }
}

if ($ActiveSuites.Count -eq 0) {
    Write-Host "[ERROR] No test suites selected for Tier: $Tier (Filter: '$Filter')"
    exit 1
}

$TotalSuites = $ActiveSuites.Count
$PassedSuites = 0
$FailedSuites = 0
$SuiteStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($s in $ActiveSuites) {
    $SuiteName = $s.Name
    $SourcePath = Join-Path $ScriptDir $s.Source
    $ExePath = Join-Path $BinDir "$SuiteName.exe"
    $Libs = $s.Libs

    Write-Host ""
    Write-Host "[BUILD] Compiling Tier $($s.Tier) Suite: $SuiteName"

    $CompileArgs = @(
        "-std=c++17",
        "-O2",
        "-I`"$IncludeDir`"",
        "-I`"$SrcDir`"",
        "`"$SourcePath`"",
        "-o", "`"$ExePath`""
    )
    if ($Libs.Count -gt 0) {
        $CompileArgs += $Libs
    }

    & g++ @CompileArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Compilation failed for: $SourcePath" -ForegroundColor Red
        $FailedSuites++
        continue
    }

    Write-Host "[EXEC]  Running $SuiteName..."
    $RunArgs = @()
    if ($Filter -ne "" -and -not $SuiteMatchFilter) {
        $RunArgs += "--filter=$Filter"
    }
    if ($VerboseOutput) {
        $RunArgs += "--verbose"
    }

    if ($RunArgs.Count -gt 0) {
        & $ExePath @RunArgs
    } else {
        & $ExePath
    }

    if ($LASTEXITCODE -eq 0) {
        $PassedSuites++
    } else {
        $FailedSuites++
        Write-Host "[FAIL] Suite failed: $SuiteName (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
    }
}

$SuiteStopwatch.Stop()
$TotalElapsed = [math]::Round($SuiteStopwatch.Elapsed.TotalMilliseconds, 2)

Write-Host ""
Write-Host "========================================================================="
Write-Host "[FINAL SUMMARY] Master Test Suite Execution Complete"
Write-Host "  Suites Executed : $TotalSuites"
Write-Host "  Suites Passed   : $PassedSuites"
Write-Host "  Suites Failed   : $FailedSuites"
Write-Host "  Total Runtime   : $TotalElapsed ms"
Write-Host "========================================================================="

if ($FailedSuites -eq 0) {
    Write-Host "[STATUS] ALL TEST SUITES PASSED (100% INTEGRITY GUARANTEED)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "[STATUS] TEST SUITE RUN REPORTED FAILURES" -ForegroundColor Red
    exit 1
}
