# Verify and rebuild the current CPC bundle without switching refs or publishing tags.
param([string]$Root = "", [switch]$SkipTests, [switch]$SkipLatex)
$ErrorActionPreference = "Stop"
if (-not $Root) {
    if ($env:BECGPP_CPC_ROOT) { $Root = $env:BECGPP_CPC_ROOT }
    else {
        $candidate = Get-Item -LiteralPath $PSScriptRoot
        while ($candidate -and -not (Test-Path -LiteralPath (Join-Path $candidate.FullName 'latex\main.tex'))) { $candidate = $candidate.Parent }
        if ($candidate) { $Root = $candidate.FullName }
    }
}
if (-not $Root -or -not (Test-Path -LiteralPath (Join-Path $Root 'latex\main.tex'))) { throw 'Set -Root to the directory containing code, data and latex.' }
$Root = (Resolve-Path -LiteralPath $Root).Path
$env:BECGPP_CPC_ROOT = $Root
$Repo = Join-Path $Root 'code\becGPP'
function Run($exe, [string[]]$argv) { & $exe @argv; if ($LASTEXITCODE -ne 0) { throw "$exe failed ($LASTEXITCODE)" } }
# The verifier checks numerical/test hashes against the immutable release cell.
# Preserve all local edits; never checkout --force, create or push a tag here.
if (-not $SkipTests) {
    $RunOut = Join-Path $Root ('data\review\local_release_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    New-Item -ItemType Directory -Path $RunOut -ErrorAction Stop | Out-Null
    $oldCuda = $env:CUDA_VISIBLE_DEVICES
    $env:CUDA_VISIBLE_DEVICES = ''
    Push-Location -LiteralPath $Repo
    try {
        Run python @('-m','pytest','-q',('--junitxml=' + (Join-Path $RunOut 'cpu_suite.xml')))
        Run python @('examples/cpu_test_run.py','--out',(Join-Path $RunOut 'reference'),'--check')
    } finally {
        Pop-Location
        if ($null -eq $oldCuda) { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue } else { $env:CUDA_VISIBLE_DEVICES = $oldCuda }
    }
}
if (-not $SkipLatex) {
    $Latex = Join-Path $Root 'latex'
    New-Item -ItemType Directory -Force -Path (Join-Path $Latex 'build') | Out-Null
    Push-Location -LiteralPath $Latex
    try {
        Run pdflatex @('-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex')
        Run bibtex @('build/main')
        Run pdflatex @('-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex')
        Run pdflatex @('-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex')
        Copy-Item -LiteralPath 'build\main.pdf' -Destination 'main.pdf' -Force
    } finally { Pop-Location }
}
Run python @((Join-Path $Root 'code\tools\verify_final_project.py'))
Run python @((Join-Path $Root 'code\tools\prepare_final_submission.py'))
Write-Host 'Archives and SHA256.json rebuilt. Read the evidence status and author actions before submission.'
