<#
  Removes the ComfyUI memory watchdog: unregisters the scheduled task and deletes %LOCALAPPDATA%\memory-watchdog
  (the installed script, config.json and the log). Docker, WSL and .wslconfig are left as they are.

    powershell -ExecutionPolicy Bypass -File tools\windows-memory-watchdog\uninstall.ps1
#>
$TaskName = 'ComfyUI memory watchdog'
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Remove-Item (Join-Path $env:LOCALAPPDATA 'memory-watchdog') -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Removed '$TaskName'."
