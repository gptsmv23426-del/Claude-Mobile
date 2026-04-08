# Run this script ONCE as Administrator to register the bot as an auto-start task.
# Right-click PowerShell -> "Run as Administrator", then paste:
#   cd "C:\Users\gvsmv\CLAUDE\Github\Trading-Bot"
#   .\register_autostart.ps1

$taskName  = "PolymarketBot"
$scriptDir = "C:\Users\gvsmv\CLAUDE\Github\Trading-Bot"
$batFile   = "$scriptDir\start_bot.bat"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action  = New-ScheduledTaskAction -Execute $batFile -WorkingDirectory $scriptDir
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName  $taskName `
    -Action    $action `
    -Trigger   $trigger `
    -Settings  $settings `
    -Force

Write-Host "Task '$taskName' registered. Bot will start automatically on next logon." -ForegroundColor Green
Write-Host "To start now without rebooting, run:  Start-ScheduledTask -TaskName PolymarketBot" -ForegroundColor Cyan
