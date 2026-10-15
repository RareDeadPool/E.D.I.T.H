Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

Function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

[Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager, Windows.Media.Control, ContentType = WindowsRuntime] | Out-Null
$managerOp = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync()
$manager = Await $managerOp ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager])

$session = $manager.GetCurrentSession()
if ($session) {
    $infoOp = $session.TryGetMediaPropertiesAsync()
    $info = Await $infoOp ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties])
    Write-Output "SourceApp: $($session.SourceAppUserModelId)"
    Write-Output "Title: $($info.Title)"
    Write-Output "Artist: $($info.Artist)"
} else {
    Write-Output "No active media session"
}
