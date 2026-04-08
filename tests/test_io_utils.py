"""Unit tests for filesystem and runtime helpers."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tests.test_support import import_project_module, make_dependency_modules


class IoUtilsTests(unittest.TestCase):
    """Verify the local I/O and environment helper functions."""

    def setUp(self) -> None:
        """Import the I/O helper module with dependency stubs."""
        self.tf_stub, modules = make_dependency_modules()
        self.module = import_project_module(
            "utils.io_utils",
            extra_modules=modules,
            clear_modules=("utils.io_utils",),
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
            "utils.io_utils",
            extra_modules=modules,
            clear_modules=("utils.io_utils",),
        )

        module.handle_gpu_compatibility()

        self.assertEqual(
            tf_stub.memory_growth_calls,
            [("GPU:0", True), ("GPU:1", True)],
        )

    def test_get_main_path_and_save_results_create_expected_files(self) -> None:
        """Artifact helpers should create the model directory and JSON payload."""
        with tempfile.TemporaryDirectory() as tempdir:
            current_dir = os.getcwd()
            self.addCleanup(os.chdir, current_dir)
            os.chdir(tempdir)

            main_path = self.module.get_main_path("v2")
            model_path = self.module.get_model_path(main_path, "MNIST")
            self.module.save_results_as_json(main_path, {"MNIST": {"accuracy": 0.99}})

            self.assertEqual(main_path, os.path.join("models", "v2"))
            self.assertEqual(model_path, os.path.join("models", "v2", "MNIST_model.h5"))
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


if __name__ == "__main__":
    unittest.main()
