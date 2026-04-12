@echo off
setlocal

set REPO=C:\Users\DELL\Desktop\FedOOD
set PY=C:\Users\DELL\anaconda3\envs\plankton_vis\python.exe
set LOG=C:\Users\DELL\Desktop\FedOOD\experiments\queue_final_dell_gpu0.log

if not exist "%REPO%\experiments" mkdir "%REPO%\experiments"
echo [start] %date% %time% > "%LOG%"

cd /d "%REPO%"

echo [1/2] START FOSTER resnet101 >> "%LOG%"
"%PY%" -u train_foster.py --model_type resnet101 --data_root ./Plankton_OOD_Dataset --communication_rounds 50 --local_epochs 4 --batch_size 32 --image_size 320 --n_clients 5 --alpha 0.1 --seed 42 --device cuda:0 --num_workers 8 --output_dir ./experiments/foster_manifest_final_v1 >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1
echo [1/2] END FOSTER resnet101 >> "%LOG%"

echo [2/2] START FOSTER resnet50 >> "%LOG%"
"%PY%" -u train_foster.py --model_type resnet50 --data_root ./Plankton_OOD_Dataset --communication_rounds 50 --local_epochs 4 --batch_size 32 --image_size 320 --n_clients 5 --alpha 0.1 --seed 42 --device cuda:0 --num_workers 8 --output_dir ./experiments/foster_manifest_final_v1 >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1
echo [2/2] END FOSTER resnet50 >> "%LOG%"

echo [done] %date% %time% >> "%LOG%"
