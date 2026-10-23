import importlib.util
import tempfile
import unittest
from pathlib import Path


class BackendTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
        "optional torch dependencies unavailable",
    )
    def test_train_validation_test_and_atomic_publication(self):
        from unittest.mock import patch

        import torch

        from vision_workbench.cnn import train_cifar10

        accessed = {True: set(), False: set()}

        class FakeData(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                self.train = train

            def __len__(self):
                return 20 if self.train else 4

            def __getitem__(self, index):
                accessed[self.train].add(index)
                return torch.full((3, 32, 32), index / 20), index % 10

        torch.set_num_threads(1)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("torchvision.datasets.CIFAR10", FakeData),
        ):
            target = Path(folder) / "trained.pt"
            result = train_cifar10(folder, target, epochs=2, batch_size=2, width=2)
            payload = torch.load(target, weights_only=True)
            self.assertEqual(
                payload["validation_accuracy"],
                max(row["validation_accuracy"] for row in result["history"]),
            )
            self.assertEqual(accessed[True], set(range(20)))
            self.assertEqual(accessed[False], set(range(4)))
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                train_cifar10(folder, target, epochs=1, batch_size=2, width=2)
            self.assertEqual(target.read_bytes(), original)
            failed = Path(folder) / "failed.pt"
            with (
                patch("torch.save", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                train_cifar10(folder, failed, epochs=1, batch_size=2, width=2)
            self.assertFalse(failed.exists())
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
        "optional torch dependencies unavailable",
    )
    def test_training_gradient_and_safe_checkpoint_roundtrip(self):
        import torch
        from PIL import Image

        from vision_workbench.cnn import CLASSES, build_model, predict_image

        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = build_model(2)
        x = torch.randn(2, 3, 32, 32)
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model[0].weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(
                {"state": model.state_dict(), "width": 2, "classes": CLASSES}, path
            )
            result = predict_image(Image.new("RGB", (40, 50)), path)
            self.assertEqual(len(result), 10)
            self.assertAlmostEqual(sum(r["score"] for r in result), 1, places=5)
            self.assertGreaterEqual(result[0]["score"], result[-1]["score"])


if __name__ == "__main__":
    unittest.main()
