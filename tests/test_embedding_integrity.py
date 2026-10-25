import unittest
import numpy as np
from vision_workbench.recognition import normalize_rows, build_classifier, rank


class EmbeddingIntegrityTests(unittest.TestCase):
    def test_complex_vectors_are_rejected_without_discarding_components(self):
        with self.assertRaisesRegex(ValueError, 'real'):
            normalize_rows([[1 + 2j, 1]])
        fit = build_classifier([[[1, 0]], [[0, 1]]])
        with self.assertRaisesRegex(ValueError, 'real'):
            rank([1 + 2j, 0], fit, ['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'real'):
            build_classifier([[[1, 0]]], [[1, 0]], [[[1 + 2j, 0]]])

    def test_invalid_class_indices_raise_validation_errors(self):
        fit = build_classifier([[[1, 0]], [[0, 1]]])
        for indices in ([0., 1.], [False, True], [[0], [1]], [0, 0], [-1, 1]):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, 'indices'):
                rank([1, 0], dict(fit, indices=np.asarray(indices)), ['a', 'b'])

    def test_valid_integer_index_types_keep_ranking(self):
        fit = build_classifier([[[1, 0]], [[0, 1]]])
        fit['indices'] = np.array([0, 1], dtype=np.int32)
        self.assertEqual(rank([1, 0], fit, ['a', 'b'])[0]['label'], 'a')
