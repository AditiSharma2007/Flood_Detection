#!/usr/bin/env python3
"""
Hardware acceleration diagnostic tool.
Checks availability of CUDA (NVIDIA GPU), MPS (Apple Silicon GPU), and CPU.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

print("=" * 50)
print("PYTORCH HARDWARE DIAGNOSTICS")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available:  {cuda_available}")
if cuda_available:
    print(f"  Device Name:   {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  VRAM Total:    {vram_gb:.2f} GB")

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"MPS Available:   {mps_available} (Apple Silicon GPU)")

if not cuda_available and not mps_available:
    print("Active Fallback: CPU")
print("=" * 50)
