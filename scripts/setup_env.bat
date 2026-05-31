@echo off
:: Batch script to set up virtual environment and install dependencies on Windows

set ENV_NAME=phygecnet

echo === Creating conda environment: %ENV_NAME% ===
call conda create -n %ENV_NAME% python=3.10 -y

echo === Activating conda environment ===
call conda activate %ENV_NAME%

echo === Installing PyTorch with CUDA 11.8 support ===
call pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

echo === Installing python package dependencies ===
call pip install -r requirements.txt

echo === Installing project package in editable mode ===
call pip install -e .

echo === Setup complete! ===
echo To activate this environment, run:
echo   conda activate %ENV_NAME%
pause
