"""Unit tests for evaluation and reporting helpers."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from tests.test_support import import_project_module, make_dependency_modules


class FakeModel:
    """Minimal evaluation model used by helper tests."""

    def evaluate(self, _x_test: object, _y_test: object) -> tuple[float, float]:
        """Return deterministic metrics for rounding assertions."""
        return (0.123456, 0.987654)


class EvalUtilsTests(unittest.TestCase):
    """Cover the reporting helpers without importing real plotting libraries."""

    def setUp(self) -> None:
        """Import the evaluation helper module with dependency stubs."""
        self.tf_stub, modules = make_dependency_modules()
        self.module = import_project_module(
            "utils.eval_utils",
            extra_modules=modules,
            clear_modules=("utils.eval_utils",),
        )

    def test_evaluate_rounds_metrics(self) -> None:
        """Evaluation should round loss and accuracy to five decimal places."""
        datasets = {
            "MNIST": {"x_test": [1], "y_test": [2]},
            "SVHN": {"x_test": [3], "y_test": [4]},
        }

        results = self.module.evaluate(FakeModel(), datasets, ["MNIST", "SVHN"])

        self.assertEqual(results["MNIST"]["loss"], 0.12346)
        self.assertEqual(results["SVHN"]["accuracy"], 0.98765)

    def test_print_results_outputs_dataset_metrics(self) -> None:
        """Printed summaries should include the trained-with banner and metrics."""
        output = io.StringIO()
        results = {
            "MNIST": {"loss": 0.1, "accuracy": 0.9},
        }

        with redirect_stdout(output):
            self.module.print_results(results, "MNIST", "v2")

        rendered = output.getvalue()
        self.assertIn("Trained with: MNIST model version v2", rendered)
        self.assertIn("MNIST Test accuracy: 0.9", rendered)


if __name__ == "__main__":
    unittest.main()
