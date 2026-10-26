import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from vision_workbench.recognition import evaluate_predictions


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_imbalanced_labels_and_unpredicted_class(self):
        result = evaluate_predictions(['a'] * 9 + ['b'], ['a'] * 10, ['a', 'b', 'unused'])
        self.assertAlmostEqual(result['accuracy'], .9)
        self.assertAlmostEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['per_class'][1]['recall'], 0)
        self.assertEqual(result['per_class'][2]['support'], 0)
        self.assertAlmostEqual(result['macro_f1'], (18 / 19) / 3)
        self.assertEqual(result['errors'], [dict(expected='b', predicted='a', count=1)])

    def test_ui_report_survives_rerun_but_hides_after_edit(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        next(b for b in app.button if b.label == 'Evaluate labelled predictions').click().run()
        self.assertFalse(app.exception)
        app.run()
        self.assertTrue(any(m.label == 'Macro F1' for m in app.metric))
        next(t for t in app.text_area if t.label == 'Predicted labels, one per line').set_value('cat\ndog\ncat').run()
        self.assertFalse(app.exception)
        self.assertFalse(any(m.label == 'Macro F1' for m in app.metric))
        self.assertTrue(any('Evaluation inputs changed' in i.value for i in app.info))
