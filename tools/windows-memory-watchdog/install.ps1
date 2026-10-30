<#
  Installs the ComfyUI memory watchdog on Windows: copies memory_watchdog.py to %LOCALAPPDATA%\memory-watchdog and
  registers the "ComfyUI memory watchdog" scheduled task, which runs it at logon and then every minute through
  pythonw.exe, so no window ever opens. Run it again after editing the script or moving Python.

    powershell -ExecutionPolicy Bypass -File tools\windows-memory-watchdog\install.ps1 [-Python C:\path\to\python.exe]

  Undo: uninstall.ps1 beside this file. Pause: Disable-ScheduledTask -TaskName 'ComfyUI memory watchdog'.
#>
param([string] $Python)   # default: the py launcher's default Python 3

$ErrorActionPreference = 'Stop'
$TaskName = 'ComfyUI memory watchdog'
$Dir = Join-Path $env:LOCALAPPDATA 'memory-watchdog'

if (-not $Python) { $Python = & py -3 -c 'import sys; print(sys.executable)' }
$Pythonw = Join-Path (Split-Path $Python) 'pythonw.exe'
if (-not (Test-Path $Pythonw)) { throw "No pythonw.exe beside $Python; pass -Python with a python.exe that has one." }

New-Item -ItemType Directory -Force $Dir | Out-Null
Copy-Item (Join-Path $PSScriptRoot 'memory_watchdog.py') $Dir -Force
$Script = Join-Path $Dir 'memory_watchdog.py'

$User = "$env:USERDOMAIN\$env:USERNAME"
$Action = New-ScheduledTaskAction -Execute $Pythonw -Argument "`"$Script`"" -WorkingDirectory $Dir
# Once at logon, and every minute from now on, indefinitely (a logon trigger's own repetition would wait for the next logon).
$Triggers = @(
  (New-ScheduledTaskTrigger -AtLogOn -User $User),
  (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1))
)
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
  -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -Priority 7
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Description = "Drops the docker-desktop WSL VM's file cache when Windows runs short of RAM, so ComfyUI model loads " +
  "do not stall, and once ComfyUI has been idle for 10 minutes. Never stops ComfyUI, Docker or WSL; runs through pythonw.exe, so no window opens. " +
  "Log: $Dir\watchdog.log. Undo: Unregister-ScheduledTask -TaskName '$TaskName'."

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Triggers -Settings $Settings -Principal $Principal `
  -Description $Description -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Installed '$TaskName': $Pythonw $Script, every minute from logon. Log: $Dir\watchdog.log"
& $Python $Script --status
