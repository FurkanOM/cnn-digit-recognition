"""Unit tests for the shared training helper module."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
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


class FakeModel:
    """Minimal evaluation model used by helper tests."""

    def evaluate(self, _x_test: object, _y_test: object) -> tuple[float, float]:
        """Return deterministic metrics for rounding assertions."""
        return (0.123456, 0.987654)


class TrainingHelpersTests(unittest.TestCase):
    """Cover the project helper functions without importing real ML libraries."""

    def setUp(self) -> None:
        """Import the helper module with dependency stubs."""
        self.tf_stub, modules = make_dependency_modules()
        self.module = import_project_module(
            "helpers",
            extra_modules=modules,
            clear_modules=("helpers",),
        )

    def test_handle_args_parses_gpu_flag_and_version(self) -> None:
        """The CLI parser should accept the historical GPU flag and version value."""
        args = self.module.handle_args(["-handle-gpu", "--version", "v1"])

        self.assertTrue(args.handle_gpu)
        self.assertEqual(args.version, "v1")

    def test_handle_gpu_compatibility_enables_memory_growth_for_each_gpu(self) -> None:
        """GPU compatibility should be enabled once per visible GPU."""
        tf_stub, modules = make_dependency_modules(gpus=["GPU:0", "GPU:1"])
        module = import_project_module(
            "helpers",
            extra_modules=modules,
            clear_modules=("helpers",),
        )

        module.handle_gpu_compatibility()

        self.assertEqual(
            tf_stub.memory_growth_calls,
            [("GPU:0", True), ("GPU:1", True)],
        )

    def test_get_input_shape_uses_configured_image_layout(self) -> None:
        """The helper should mirror the configured Keras backend layout."""
        channels_first_stub, modules = make_dependency_modules(image_data_format="channels_first")
        _ = channels_first_stub
        channels_first_module = import_project_module(
            "helpers",
            extra_modules=modules,
            clear_modules=("helpers",),
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

    def test_get_main_path_and_save_results_create_expected_files(self) -> None:
        """Artifact helpers should create the model directory and JSON payload."""
        with tempfile.TemporaryDirectory() as tempdir:
            current_dir = os.getcwd()
            self.addCleanup(os.chdir, current_dir)
            os.chdir(tempdir)

            main_path = self.module.get_main_path("v2")
            self.module.save_results_as_json(main_path, {"MNIST": {"accuracy": 0.99}})

            self.assertEqual(main_path, os.path.join("models", "v2"))
            self.assertTrue(os.path.isdir(main_path))
            with open(os.path.join(main_path, "results.json"), "r", encoding="utf-8") as file_obj:
                self.assertEqual(json.load(file_obj), {"MNIST": {"accuracy": 0.99}})

    def test_get_files_returns_nested_file_metadata(self) -> None:
        """Directory traversal should include nested files with filenames."""
        with tempfile.TemporaryDirectory() as tempdir:
            nested_dir = os.path.join(tempdir, "nested")
            os.makedirs(nested_dir)
            file_paths = [
                os.path.join(tempdir, "top.txt"),
                os.path.join(nested_dir, "child.txt"),
            ]
            for path in file_paths:
                with open(path, "w", encoding="utf-8") as file_obj:
                    file_obj.write("content")

            files = self.module.get_files(tempdir)

        self.assertCountEqual(
            files,
            [
                {"path": file_paths[0], "filename": "top.txt"},
                {"path": file_paths[1], "filename": "child.txt"},
            ],
        )

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

        with mock.patch.object(self.module, "get_MNIST_data", return_value=mnist_split), mock.patch.object(
            self.module,
            "get_SVHN_data",
            return_value=svhn_split,
        ):
            datasets = self.module.get_datasets(["MNIST", "SVHN"], n_combinations=2)

        self.assertEqual(datasets["MNIST"], mnist_split)
        self.assertEqual(datasets["SVHN"], svhn_split)
        self.assertEqual(datasets["MNIST+SVHN"]["x_train"], [1, 7])

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

    def test_legacy_convert_function_alias_is_preserved(self) -> None:
        """The historical conversion helper name should still resolve to the new one."""
        self.assertIs(
            self.module.convert_array_to_MNIST_type_img,
            self.module.convert_array_to_mnist_type_image,
        )


if __name__ == "__main__":
    unittest.main()
