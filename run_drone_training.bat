@echo off
title MobileNetV2 Drone and Hard Negatives Training
color 0A
echo =====================================================================
echo   STARTING FLOOD DETECTION TRAINING ON DRONE + HARD NEGATIVES
echo =====================================================================
echo.
cd /d "%~dp0"
python -u train_drone_and_hard_negatives.py
echo.
echo =====================================================================
echo   TRAINING COMPLETE!
echo =====================================================================
