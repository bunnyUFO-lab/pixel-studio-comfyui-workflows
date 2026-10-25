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

# The user Path with $Bin appended, or $null when it is already there. $Raw is the unexpanded registry value; entries
# are compared in their expanded form, case-insensitively, ignoring a trailing backslash. Existing entries are kept verbatim.
function Get-UserPathUpdate([string]$Raw, [string]$Bin) {
    $want = $Bin.TrimEnd("\")
    $entries = @()
    if ($Raw) { $entries = @($Raw -split ";" | Where-Object { $_ }) }
    foreach ($e in $entries) {
        if ([Environment]::ExpandEnvironmentVariables($e).TrimEnd("\") -ieq $want) { return $null }
    }
    return (@($entries) + $Bin) -join ";"
}

# A registry write does not tell running programs; broadcast WM_SETTINGCHANGE so new Explorer-launched terminals see it.
function Send-EnvironmentChange {
    try {
        Add-Type -Namespace Psc -Name Native -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Auto)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint flags, uint timeout, out System.UIntPtr result);
"@
        $out = [UIntPtr]::Zero
        [void][Psc.Native]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, "Environment", 2, 5000, [ref]$out)
    } catch {
        Write-Host "NOTE: could not notify running programs; sign out and in to see the new PATH everywhere."
    }
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
    $envKey = Get-Item -LiteralPath "HKCU:\Environment"
    # The raw value keeps %VAR% references and the registry kind (REG_EXPAND_SZ); GetEnvironmentVariable would flatten both.
    $raw = [string]$envKey.GetValue("Path", "", [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
    if ($envKey.GetValueNames() -contains "Path") { $kind = $envKey.GetValueKind("Path") }
    $update = Get-UserPathUpdate $raw $bin
    if ($null -eq $update) {
        Write-Host "$bin is already on your user PATH."
    } else {
        Set-ItemProperty -LiteralPath "HKCU:\Environment" -Name "Path" -Value $update -Type $kind
        Send-EnvironmentChange
        Write-Host "Added $bin to your user PATH. Open a new terminal to use: pixel-studio-comfy doctor"
    }
    $want = $bin.TrimEnd("\")
    if (($env:Path -split ";" | ForEach-Object { $_.TrimEnd("\") }) -inotcontains $want) { $env:Path = "$($env:Path);$bin" }
}

exit $code
