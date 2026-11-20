"""
Unit tests for inference and multi-crop transform pipeline.
"""

import unittest
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_flood_detector, get_device
from src.transforms import make_crops, get_eval_transform
from predict import predict_single

class TestInference(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        self.model = load_flood_detector(device=self.device)
        self.test_img_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test_images", "Screenshot 2026-10-31 033726.png"
        )

    def test_make_crops(self):
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        crops = make_crops(img)
        self.assertEqual(len(crops), 8)
        for c in crops:
            self.assertEqual(c.shape, (3, 224, 224))

    def test_predict_single_output_contract(self):
        self.assertTrue(os.path.exists(self.test_img_path), f"Test image not found at {self.test_img_path}")
        res = predict_single(self.model, self.test_img_path, self.device)

        self.assertIn("prediction", res)
        self.assertIn("status", res)
        self.assertIn("confidence_pct", res)
        self.assertIn("flood_prob_pct", res)
        self.assertIn("normal_prob_pct", res)

        total_prob = res["flood_prob_pct"] + res["normal_prob_pct"]
        self.assertAlmostEqual(total_prob, 100.0, places=1)
        self.assertIn(res["prediction"], ["Flooding", "Normal"])

if __name__ == "__main__":
    unittest.main()
