@echo off
setlocal

set REPO=C:\Users\DELL\Desktop\FedOOD
set PY=C:\Users\DELL\anaconda3\envs\plankton_vis\python.exe
set LOG=C:\Users\DELL\Desktop\FedOOD\experiments\queue_final_dell_gpu1.log

if not exist "%REPO%\experiments" mkdir "%REPO%\experiments"
echo [start] %date% %time% > "%LOG%"

cd /d "%REPO%"

echo [1/2] START FOSTER densenet169 >> "%LOG%"
"%PY%" -u train_foster.py --model_type densenet169 --data_root ./Plankton_OOD_Dataset --communication_rounds 50 --local_epochs 4 --batch_size 32 --image_size 320 --n_clients 5 --alpha 0.1 --seed 42 --device cuda:1 --num_workers 8 --output_dir ./experiments/foster_manifest_final_v1 >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1
echo [1/2] END FOSTER densenet169 >> "%LOG%"

echo [2/2] START FOSTER efficientnet_v2_s >> "%LOG%"
"%PY%" -u train_foster.py --model_type efficientnet_v2_s --data_root ./Plankton_OOD_Dataset --communication_rounds 50 --local_epochs 4 --batch_size 32 --image_size 320 --n_clients 5 --alpha 0.1 --seed 42 --device cuda:1 --num_workers 8 --output_dir ./experiments/foster_manifest_final_v1 >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1
echo [2/2] END FOSTER efficientnet_v2_s >> "%LOG%"

echo [done] %date% %time% >> "%LOG%"
