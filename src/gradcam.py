"""
Grad-CAM Explainability Engine for MobileNetV2.
"""

import os
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    using PyTorch hooks on a convolutional target layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None
        self.fwd_hook = self.target_layer.register_forward_hook(self._fwd_hook_fn)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._bwd_hook_fn)

    def _fwd_hook_fn(self, module, inp, output):
        self.activation = output

    def _bwd_hook_fn(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generates CAM heatmap for given input tensor (1, 3, H, W).
        Returns:
            cam: 2D numpy array [0, 1]
            probs: 1D torch.Tensor of class probabilities
            class_idx: target predicted class index
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradient[0].cpu().detach().numpy()
        activations = self.activation[0].cpu().detach().numpy()

        # Global average pooling over gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val

        return cam, probs, class_idx

    def close(self):
        """Removes registered hooks."""
        if self.fwd_hook:
            self.fwd_hook.remove()
        if self.bwd_hook:
            self.bwd_hook.remove()

def make_cam_overlay(orig_img: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    """
    Overlays Grad-CAM heatmap onto the original PIL image.
    """
    w, h = orig_img.size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_norm = np.array(cam_pil) / 255.0

    cmap = matplotlib.colormaps['jet']
    heatmap_rgba = cmap(cam_norm)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    overlay_pil = Image.blend(orig_img, heatmap_pil, alpha=alpha)
    return cam_norm, overlay_pil
