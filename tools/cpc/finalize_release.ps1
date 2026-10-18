# Bring the CPC bundle to release v1.2.2 and rebuild the submission archives.
#   powershell -ExecutionPolicy Bypass -File D:\CPC\code\tools\finalize_release.ps1
# Options: -Root D:\CPC  -SkipTests  -SkipLatex
param(
    [string]$Root = "",
    [switch]$SkipTests,
    [switch]$SkipLatex
)
$ErrorActionPreference = "Stop"
$Tag = "v1.2.2"
$TagCommit = "1c7eef613d147b6857375f9a61342b38cae1bc86"   # verified by GitHub CI (109 tests, reference run)
if (-not $Root) {
    if ($env:BECGPP_CPC_ROOT) { $Root = $env:BECGPP_CPC_ROOT }
    else { $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
}
if (-not (Test-Path (Join-Path $Root "latex\main.tex"))) { throw "Bundle not found at $Root (expected latex\main.tex)" }
$env:BECGPP_CPC_ROOT = $Root
$Repo = Join-Path $Root "code\becGPP"
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) { & $exe @argv; if ($LASTEXITCODE -ne 0) { throw "$exe $($argv -join ' ') failed ($LASTEXITCODE)" } }

Step "1. Create (if needed) and check out $Tag in code\becGPP"
Run git @("-C", $Repo, "fetch", "--tags", "origin")
$remoteTag = git -C $Repo ls-remote --tags origin "refs/tags/$Tag"
if (-not $remoteTag) {
    $localTag = git -C $Repo tag --list $Tag
    if (-not $localTag) {
        Run git @("-C", $Repo, "tag", "-a", $Tag, $TagCommit, "-m", "becGPP 1.2.2 - release described in the Computer Physics Communications article")
    }
    Run git @("-C", $Repo, "push", "origin", $Tag)
}
$tagged = (git -C $Repo rev-parse "$Tag^{commit}").Trim()
if ($tagged -ne $TagCommit) { throw "$Tag points to $tagged, expected $TagCommit" }
Run git @("-C", $Repo, "checkout", "--force", $Tag)
$untracked = git -C $Repo status --porcelain
if ($untracked) { Write-Host $untracked; throw "code\becGPP is not clean after checkout; move the files above out of the repository and rerun." }
Write-Host ("HEAD = " + (git -C $Repo rev-parse HEAD))

Step "2. Install the release"
Run python @("-m", "pip", "install", "-e", "$Repo[yaml,test]")

if (-not $SkipTests) {
    Step "3. CPU test suite and reference test run"
    Push-Location $Repo
    try {
        Run python @("-m", "pytest", "-q")
        Run python @("examples\cpu_test_run.py", "--out", (Join-Path $Root "data\raw\release_1_2_2_local_cpu"), "--check")
    } finally { Pop-Location }
}

if (-not $SkipLatex) {
    Step "4. Compile the manuscript"
    $Latex = Join-Path $Root "latex"
    New-Item -ItemType Directory -Force -Path (Join-Path $Latex "build") | Out-Null
    Push-Location $Latex
    try {
        Run pdflatex @("-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex")
        Run bibtex @("build/main")
        Run pdflatex @("-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex")
        Run pdflatex @("-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex")
        Copy-Item -LiteralPath "build\main.pdf" -Destination "main.pdf" -Force
    } finally { Pop-Location }
}

Step "5. Verify the bundle"
Run python @((Join-Path $Root "code\tools\verify_final_project.py"))

Step "6. Rebuild the submission archives"
Run python @((Join-Path $Root "code\tools\prepare_final_submission.py"))
Write-Host "`nDone. Archives and SHA256.json are in $Root\data\submission" -ForegroundColor Green
