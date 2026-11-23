"""
Unit tests for Flood Detection model architecture and weight loading.
"""

import unittest
import torch
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_mobilenet_v2, load_flood_detector, get_device

class TestModel(unittest.TestCase):
    def test_model_architecture(self):
        model = build_mobilenet_v2()
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 2))

    def test_device_selection(self):
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_load_production_checkpoint(self):
        model = load_flood_detector()
        self.assertIsNotNone(model)
        self.assertFalse(model.training)

if __name__ == "__main__":
    unittest.main()
