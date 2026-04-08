"""Unit tests for model-construction helpers."""

from __future__ import annotations

import unittest

from tests.test_support import import_project_module, make_dependency_modules


class ModelUtilsTests(unittest.TestCase):
    """Verify model dispatch and architecture construction behavior."""

    def setUp(self) -> None:
        """Import the model helper module with dependency stubs."""
        self.tf_stub, modules = make_dependency_modules()
        self.module = import_project_module(
            "utils.model_utils",
            extra_modules=modules,
            clear_modules=("utils.model_utils",),
        )

    def test_get_model_dispatches_to_supported_versions(self) -> None:
        """Model builders should remain available through the version dispatcher."""
        model_v1 = self.module.get_model((28, 28, 1), "v1")
        model_v2 = self.module.get_model((28, 28, 1), "v2")

        self.assertEqual(len(model_v1.layers), 7)
        self.assertEqual(model_v1.layers[0]["layer"], "Conv2D")
        self.assertEqual(len(model_v2.layers), 18)
        self.assertEqual(model_v2.layers[-1]["layer"], "Dropout")

    def test_get_model_rejects_unknown_versions(self) -> None:
        """Unsupported model versions should keep raising an explicit exception."""
        with self.assertRaisesRegex(Exception, "Please give a valid version for training model"):
            self.module.get_model((28, 28, 1), "v3")


if __name__ == "__main__":
    unittest.main()
