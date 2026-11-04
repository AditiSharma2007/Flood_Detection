@echo off
title MobileNetV2 Drone and Hard Negatives Training
color 0A
echo =====================================================================
echo   STARTING FLOOD DETECTION TRAINING ON DRONE + HARD NEGATIVES
echo =====================================================================
echo.
cd /d "c:\Users\Aditi Sharma\Downloads\Flood_Detection"
"D:\Flood_Detection_Data\venv\Scripts\python.exe" -u train_drone_and_hard_negatives.py
echo.
echo =====================================================================
echo   TRAINING COMPLETE!
echo =====================================================================
