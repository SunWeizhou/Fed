param(
    [Parameter(Mandatory = $true)]
    [string]$Pattern,
    [Parameter(Mandatory = $true)]
    [string]$WorkingDir,
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [Parameter(Mandatory = $true)]
    [string]$CudaVisibleDevices,
    [Parameter(Mandatory = $true)]
    [string]$ModelType,
    [Parameter(Mandatory = $true)]
    [string]$BatchSize
)

while ($true) {
    $proc = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $Pattern }
    if (-not $proc) {
        break
    }
    Start-Sleep -Seconds 60
}

Set-Location $WorkingDir
$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices

& $PythonExe train_foogd.py `
    --model_type $ModelType `
    --batch_size $BatchSize `
    --data_root ./Plankton_OOD_Dataset `
    --device cuda:0 `
    --output_dir experiments/foogd_v1
