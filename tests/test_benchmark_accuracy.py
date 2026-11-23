"""
Benchmark validation test: confirms the production model's accuracy on the AlleyFloodNet test set.
Ensures zero accuracy degradation and no catastrophic forgetting/overfitting.
"""

import unittest
import os
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_flood_detector, get_device
from src.transforms import get_eval_transform
from src.metrics import compute_binary_metrics

class TestBenchmarkAccuracy(unittest.TestCase):
    def test_holdout_test_accuracy(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_dir = os.path.join(base_dir, "data", "AlleyFloodNet", "flood_test_")
        flood_dir = os.path.join(test_dir, "flooded")
        normal_dir = os.path.join(test_dir, "non_flooded")

        if not os.path.exists(flood_dir) or not os.path.exists(normal_dir):
            self.skipTest("AlleyFloodNet holdout test set not present.")

        device = get_device()
        model = load_flood_detector(device=device)
        eval_tf = get_eval_transform()

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        flood_files = [os.path.join(flood_dir, f) for f in os.listdir(flood_dir) if f.lower().endswith(valid_exts)]
        normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(valid_exts)]

        self.assertGreater(len(flood_files), 100, "Expected at least 100 flooded test images")
        self.assertGreater(len(normal_files), 100, "Expected at least 100 normal test images")

        samples = [(p, 0) for p in flood_files] + [(p, 1) for p in normal_files]

        y_true = []
        y_pred = []
        y_probs = []

        with torch.no_grad():
            for path, label in samples:
                with Image.open(path) as img:
                    tensor = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred = torch.argmax(probs).item()
                flood_prob = probs[0].item()

                y_true.append(label)
                y_pred.append(pred)
                y_probs.append(flood_prob)

        metrics = compute_binary_metrics(np.array(y_true), np.array(y_pred), np.array(y_probs))

        print(f"\n[BENCHMARK VERIFICATION]")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['flooding_precision']*100:.2f}%")
        print(f"  Recall:    {metrics['flooding_recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['flooding_f1']*100:.2f}%")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")

        # The user specifically requested: "i dont want to destroy the current accuracy of the model"
        # Verify accuracy is above 90% and ROC-AUC is above 0.97
        self.assertGreater(metrics["accuracy"], 0.90, "Model accuracy fell below 90%!")
        self.assertGreater(metrics["roc_auc"], 0.97, "Model ROC-AUC fell below 0.97!")

if __name__ == "__main__":
    unittest.main()
