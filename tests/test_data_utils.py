"""Unit tests for dataset loading and preprocessing helpers."""

from __future__ import annotations

import unittest
from unittest import mock

from tests.test_support import import_project_module, make_dependency_modules


class ShapeOnlyArray:
    """Simple array-like object with a configurable shape and reshape recorder."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Store the shape and initialize reshape bookkeeping."""
        self.shape = shape
        self.reshape_calls = []

    def reshape(self, *shape: int) -> tuple[int, ...]:
        """Record the requested reshape and return the new shape tuple."""
        self.reshape_calls.append(shape)
        return shape


class DataUtilsTests(unittest.TestCase):
    """Verify dataset helpers without loading real ML dependencies."""

    def setUp(self) -> None:
        """Import the data helper module with dependency stubs."""
        self.tf_stub, modules = make_dependency_modules()
        self.module = import_project_module(
            "utils.data_utils",
            extra_modules=modules,
            clear_modules=("utils.data_utils",),
        )

    def test_get_input_shape_uses_configured_image_layout(self) -> None:
        """The helper should mirror the configured Keras backend layout."""
        channels_first_stub, modules = make_dependency_modules(image_data_format="channels_first")
        _ = channels_first_stub
        channels_first_module = import_project_module(
            "utils.data_utils",
            extra_modules=modules,
            clear_modules=("utils.data_utils",),
        )

        self.assertEqual(self.module.get_input_shape(), (28, 28, 1))
        self.assertEqual(channels_first_module.get_input_shape(), (1, 28, 28))

    def test_handle_channel_reshapes_using_backend_layout(self) -> None:
        """Input tensors should gain a trailing channel dimension by default."""
        array = ShapeOnlyArray((5, 28, 28))

        reshaped = self.module.handle_channel(array)

        self.assertEqual(reshaped, (5, 28, 28, 1))
        self.assertEqual(array.reshape_calls, [(5, 28, 28, 1)])

    def test_get_batch_size_follows_size_thresholds(self) -> None:
        """Batch-size thresholds should remain stable for existing workloads."""
        test_cases = [
            (4999, 2),
            (5000, 16),
            (24999, 16),
            (25000, 128),
            (49999, 128),
            (50000, 256),
            (249999, 256),
            (250000, 512),
            (499999, 512),
            (500000, 1024),
        ]

        for length, expected in test_cases:
            with self.subTest(length=length):
                train_data = ShapeOnlyArray((length,))
                self.assertEqual(self.module.get_batch_size(train_data), expected)

    def test_get_combined_dataset_concatenates_each_split(self) -> None:
        """Combined datasets should merge every train, validation, and test split."""
        datasets = {
            "MNIST": {
                "x_train": [1],
                "y_train": [2],
                "x_valid": [3],
                "y_valid": [4],
                "x_test": [5],
                "y_test": [6],
            },
            "SVHN": {
                "x_train": [7],
                "y_train": [8],
                "x_valid": [9],
                "y_valid": [10],
                "x_test": [11],
                "y_test": [12],
            },
        }

        combined = self.module.get_combined_dataset(datasets, ("MNIST", "SVHN"))

        self.assertEqual(
            combined,
            {
                "x_train": [1, 7],
                "y_train": [2, 8],
                "x_valid": [3, 9],
                "y_valid": [4, 10],
                "x_test": [5, 11],
                "y_test": [6, 12],
            },
        )

    def test_get_datasets_builds_requested_combinations(self) -> None:
        """Requested datasets should include generated combination keys."""
        mnist_split = {
            "x_train": [1],
            "y_train": [2],
            "x_valid": [3],
            "y_valid": [4],
            "x_test": [5],
            "y_test": [6],
        }
        svhn_split = {
            "x_train": [7],
            "y_train": [8],
            "x_valid": [9],
            "y_valid": [10],
            "x_test": [11],
            "y_test": [12],
        }

        with mock.patch.object(self.module, "get_mnist_data", return_value=mnist_split), mock.patch.object(
            self.module,
            "get_svhn_data",
            return_value=svhn_split,
        ):
            datasets = self.module.get_datasets(["MNIST", "SVHN"], n_combinations=2)

        self.assertEqual(datasets["MNIST"], mnist_split)
        self.assertEqual(datasets["SVHN"], svhn_split)
        self.assertEqual(datasets["MNIST+SVHN"]["x_train"], [1, 7])

    def test_legacy_aliases_are_preserved(self) -> None:
        """Legacy function names should continue pointing at the snake_case variants."""
        self.assertIs(self.module.get_MNIST_data, self.module.get_mnist_data)
        self.assertIs(
            self.module.prepare_ORHD_to_MNIST_format,
            self.module.prepare_orhd_to_mnist_format,
        )


if __name__ == "__main__":
    unittest.main()
