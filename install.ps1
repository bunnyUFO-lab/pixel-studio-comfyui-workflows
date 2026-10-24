# Installer for Windows (PowerShell 5.1 or newer, Docker Desktop with WSL2): finds or clones the repository, runs
# `pixel-studio-comfy setup`, and puts the command on PATH. All checks and file writes are in the Python CLI (psc\).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Dir PATH] [-Ref BRANCH] [-NoPath] [setup options, e.g. --preset svg --yes]
param(
    [string]$Dir,
    [string]$Ref = "main",
    [switch]$NoPath,
    [Parameter(ValueFromRemainingArguments = $true)]$Rest
)

$RepoUrl = "https://github.com/bunnyUFO-lab/pixel-studio-comfyui-workflows"

function Test-Checkout([string]$Path) {
    return (Test-Path -LiteralPath (Join-Path $Path "docker-compose.yml")) -and (Test-Path -LiteralPath (Join-Path $Path "psc"))
}

# 1. Docker first: nothing else works without it.
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "error: docker was not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
}
$os = (& docker info --format "{{.OperatingSystem}}" 2>$null | Out-String).Trim()
if ($os -notmatch "Docker Desktop") {
    Write-Host "WARN: docker info does not mention Docker Desktop (got: '$os'). Continuing; setup checks the daemon and the GPU."
}

# 2. The folder: this checkout, or a fresh clone.
if (-not $Dir) {
    if (Test-Checkout $PSScriptRoot) { $Dir = $PSScriptRoot } else { $Dir = Join-Path $HOME "pixel-studio-comfy" }
}
if (Test-Checkout $Dir) {
    Write-Host "Using the checkout in $Dir"
} elseif (Test-Path -LiteralPath $Dir) {
    Write-Host "error: $Dir exists but is not a pixel-studio-comfyui-workflows checkout. Choose another -Dir."
    exit 1
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "error: git is needed to download the repository. Install it (winget install Git.Git), then run this again."
        exit 1
    }
    Write-Host "Cloning $RepoUrl ($Ref) into $Dir"
    & git clone --branch $Ref $RepoUrl $Dir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$Dir = (Resolve-Path -LiteralPath $Dir).Path

# 3. Python 3.9+: the py launcher, then python. The Microsoft Store alias exists but fails the version test.
$pyExe = $null
$pyArgs = @()
$candidates = @(
    @{ Exe = "py"; Extra = @("-3") },
    @{ Exe = "python"; Extra = @() }
)
foreach ($cand in $candidates) {
    if (-not (Get-Command $cand.Exe -ErrorAction SilentlyContinue)) { continue }
    $extra = @($cand.Extra)
    & $cand.Exe @extra -c "import sys; sys.exit(sys.version_info < (3, 9))" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $pyExe = $cand.Exe; $pyArgs = $extra; break }
}
if (-not $pyExe) {
    Write-Host "error: Python 3.9 or newer was not found. Install it with:  winget install Python.Python.3.12"
    exit 1
}

# 4. Setup. $Rest is $null when nothing is left over; --port 8189 arrives as two strings.
$forward = @()
if ($Rest) { $forward = @($Rest | ForEach-Object { [string]$_ }) }
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Dir;$($env:PYTHONPATH)" } else { $Dir }
& $pyExe @pyArgs -m psc --dir $Dir setup @forward
$code = $LASTEXITCODE

# 5. The command on PATH (user level), unless -NoPath.
if (-not $NoPath) {
    $bin = Join-Path $Dir "bin"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $entries = @()
    if ($userPath) { $entries = @($userPath -split ";" | Where-Object { $_ }) }
    $want = $bin.TrimEnd("\")
    $present = $false
    foreach ($e in $entries) { if ($e.TrimEnd("\") -ieq $want) { $present = $true } }
    if (-not $present) {
        $new = (@($entries) + $bin) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $new, "User")
        Write-Host "Added $bin to your user PATH. Open a new terminal to use: pixel-studio-comfy doctor"
    } else {
        Write-Host "$bin is already on your user PATH."
    }
    if (($env:Path -split ";" | ForEach-Object { $_.TrimEnd("\") }) -inotcontains $want) { $env:Path = "$($env:Path);$bin" }
}

exit $code
