"""
Core model definitions and device management for Flood Detection.
"""

import os
import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = {
    0: "Flooding",
    1: "Normal"
}

def get_device(preference: str = "auto") -> torch.device:
    """
    Select the optimal available device.
    Supports 'cuda', 'mps' (Apple Silicon), or 'cpu'.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    
    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_mobilenet_v2() -> nn.Module:
    """
    Constructs the MobileNetV2 architecture with the custom dual-dropout classification head.
    Architecture:
      Dropout(0.4) -> Linear(1280, 64) -> ReLU() -> Dropout(0.2) -> Linear(64, 2)
    """
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2)
    )
    return model

def load_flood_detector(checkpoint_path: str = None, device: torch.device = None) -> nn.Module:
    """
    Loads the MobileNetV2 flood detector model and loads weights if checkpoint is provided.
    """
    if device is None:
        device = get_device()

    model = build_mobilenet_v2()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "flood_mobilenet_best.pt")

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model
