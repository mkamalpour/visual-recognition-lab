import unittest

import numpy as np

from vision_workbench.recognition import (
    build_classifier,
    evaluate_predictions,
    normalize_rows,
    rank,
)


class RecognitionTests(unittest.TestCase):
    def test_prompt_normalization_and_softmax(self):
        fit = build_classifier([[[100, 0], [0, 1]], [[-1, 0]]])
        np.testing.assert_allclose(fit["weights"][0], [2**-0.5, 2**-0.5])
        result = rank([1, 0], fit, ["a", "b"], top_k=2)
        self.assertEqual(result[0]["label"], "a")
        self.assertAlmostEqual(sum(r["score"] for r in result), 1)

    def test_class_specific_visual_and_empty_fallback(self):
        fit = build_classifier(
            [[[1, 0]], [[0, 1]], [[-1, 0]]],
            [[1, 0.1], [0.1, 1]],
            [[[1, 0]], [[0, 1]]],
            alpha=0,
        )
        np.testing.assert_array_equal(fit["counts"], [1, 1, 0])
        np.testing.assert_allclose(fit["weights"], [[1, 0], [0, 1], [-1, 0]])

    def test_refinement_ties_are_stable(self):
        fit = build_classifier([[[1, 0]], [[1, 0]], [[0, 1]]], [[1, 0]], keep=1)
        self.assertEqual(fit["indices"].tolist(), [0])

    def test_invalid_inputs(self):
        for bad in ([[0, 0]], [[float("nan"), 1]], [], [1, 2]):
            with self.assertRaises(ValueError):
                normalize_rows(bad)
        fit = build_classifier([[[1, 0]]])
        for temperature in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                rank([1, 0], fit, ["a"], temperature=temperature)
        with self.assertRaises(ValueError):
            build_classifier([[[1, 0]]], [[1, 0, 0]])

    def test_exact_evaluation(self):
        result = evaluate_predictions(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        self.assertEqual(result["confusion"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)

    def test_cancelling_reference_mean(self):
        fit = build_classifier([[[1, 0]], [[0, 1]]], [[1, 0], [-1, 0]], keep=2, alpha=1)
        self.assertEqual(fit["indices"].tolist(), [0, 1])

    def test_stable_norm_and_empty_classes(self):
        np.testing.assert_allclose(
            normalize_rows([[1e308, 1e308]]), [[2**-0.5, 2**-0.5]]
        )
        with self.assertRaises(ValueError):
            build_classifier([])


if __name__ == "__main__":
    unittest.main()
