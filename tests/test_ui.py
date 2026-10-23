import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class InterfaceTests(unittest.TestCase):
    def test_startup_and_label_evaluation(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py")
        ).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 4)
        next(
            b for b in app.button if b.label == "Evaluate labelled predictions"
        ).click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.value == "66.7%" for m in app.metric))
        self.assertTrue(
            next(b for b in app.button if b.label == "Recognize image").disabled
        )


if __name__ == "__main__":
    unittest.main()


class ReferenceTests(unittest.TestCase):
    def test_pixel_duplicates_and_deterministic_views(self):
        import io

        from PIL import Image

        from vision_workbench.ui_helpers import image_variants, reference_images

        def data(color, fmt):
            buffer = io.BytesIO()
            Image.new("RGB", (20, 20), color).save(buffer, format=fmt)
            return buffer.getvalue()

        with self.assertRaisesRegex(ValueError, "query pixels"):
            reference_images(data("red", "PNG"), [data("red", "BMP")])
        images = reference_images(
            data("red", "PNG"), [data("blue", "PNG"), data("blue", "BMP")]
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(
            [v.tobytes() for v in image_variants(images[0], 3)],
            [v.tobytes() for v in image_variants(images[0], 3)],
        )


class DescriptionTests(unittest.TestCase):
    def test_all_entries_validated_before_model_execution(self):
        from vision_workbench.ui_helpers import validate_descriptions

        for invalid in ([], {"cat": []}, {"unused": [42]}, {"cat": ["prompt"] * 101}):
            with self.assertRaises(ValueError):
                validate_descriptions(["cat"], invalid)
        self.assertEqual(
            validate_descriptions(["cat"], {"cat": ["a feline"]}), [["a feline"]]
        )
