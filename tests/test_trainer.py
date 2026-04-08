"""Unit tests for the top-level trainer entry point."""

from __future__ import annotations

import io
import types
import unittest
from contextlib import redirect_stdout

from tests.test_support import import_project_module, make_dependency_modules


class FakeTrainingModel:
    """Minimal model object used to observe trainer interactions."""

    def __init__(self) -> None:
        """Initialize the call record containers."""
        self.add_calls = []
        self.compile_calls = []
        self.fit_calls = []

    def add(self, layer: object) -> None:
        """Record the appended layer."""
        self.add_calls.append(layer)

    def compile(self, **kwargs: object) -> None:
        """Record compile keyword arguments."""
        self.compile_calls.append(kwargs)

    def fit(self, *args: object, **kwargs: object) -> None:
        """Record fit inputs."""
        self.fit_calls.append((args, kwargs))


class TrainerTests(unittest.TestCase):
    """Verify that the trainer entry point orchestrates helper calls safely."""

    def test_importing_trainer_does_not_start_training(self) -> None:
        """Importing the module should not execute the training loop."""
        _, modules = make_dependency_modules()
        call_log = []
        utils_package = types.ModuleType("utils")
        utils_package.__path__ = []
        common_module = types.ModuleType("utils.common")
        common_module.DatasetCollection = dict
        common_module.ImageShape = tuple
        common_module.NUM_CLASSES = 10
        data_utils_module = types.ModuleType("utils.data_utils")
        eval_utils_module = types.ModuleType("utils.eval_utils")
        io_utils_module = types.ModuleType("utils.io_utils")
        model_utils_module = types.ModuleType("utils.model_utils")

        def unexpected_call(*_args: object, **_kwargs: object) -> None:
            """Fail if import-time side effects trigger helper work."""
            call_log.append("called")

        data_utils_module.get_batch_size = unexpected_call
        data_utils_module.get_datasets = unexpected_call
        data_utils_module.get_input_shape = unexpected_call
        eval_utils_module.evaluate = unexpected_call
        eval_utils_module.print_results = unexpected_call
        io_utils_module.get_main_path = unexpected_call
        io_utils_module.get_model_path = unexpected_call
        io_utils_module.handle_args = unexpected_call
        io_utils_module.handle_gpu_compatibility = unexpected_call
        io_utils_module.save_results_as_json = unexpected_call
        model_utils_module.get_model = unexpected_call
        utils_package.common = common_module
        utils_package.data_utils = data_utils_module
        utils_package.eval_utils = eval_utils_module
        utils_package.io_utils = io_utils_module
        utils_package.model_utils = model_utils_module
        modules["utils"] = utils_package
        modules["utils.common"] = common_module
        modules["utils.data_utils"] = data_utils_module
        modules["utils.eval_utils"] = eval_utils_module
        modules["utils.io_utils"] = io_utils_module
        modules["utils.model_utils"] = model_utils_module

        import_project_module(
            "trainer",
            extra_modules=modules,
            clear_modules=("trainer",),
        )

        self.assertEqual(call_log, [])

    def test_main_runs_training_flow_and_persists_results(self) -> None:
        """The trainer should build, fit, evaluate, and persist each dataset run."""
        _, modules = make_dependency_modules()
        fake_model = FakeTrainingModel()
        call_log = {
            "gpu": 0,
            "print_results": [],
            "saved": None,
        }
        utils_package = types.ModuleType("utils")
        utils_package.__path__ = []
        common_module = types.ModuleType("utils.common")
        common_module.DatasetCollection = dict
        common_module.ImageShape = tuple
        common_module.NUM_CLASSES = 10
        data_utils_module = types.ModuleType("utils.data_utils")
        eval_utils_module = types.ModuleType("utils.eval_utils")
        io_utils_module = types.ModuleType("utils.io_utils")
        model_utils_module = types.ModuleType("utils.model_utils")
        datasets = {
            "MNIST": {
                "x_train": types.SimpleNamespace(shape=(10,)),
                "y_train": [1],
                "x_valid": [2],
                "y_valid": [3],
            }
        }

        data_utils_module.get_batch_size = lambda x_train: 16
        data_utils_module.get_datasets = lambda use_datasets, n_combinations=1: datasets
        data_utils_module.get_input_shape = lambda: (28, 28, 1)
        eval_utils_module.evaluate = lambda model, dataset_map, use_datasets: {
            "MNIST": {"loss": 0.1, "accuracy": 0.9}
        }
        eval_utils_module.print_results = (
            lambda results, trained_with, version: call_log["print_results"].append(
                (results, trained_with, version)
            )
        )
        io_utils_module.get_main_path = lambda version: "models/" + version
        io_utils_module.get_model_path = lambda main_path, trained_with: main_path + "/" + trained_with + ".h5"
        io_utils_module.handle_args = lambda argv=None: types.SimpleNamespace(handle_gpu=True, version="v1")
        io_utils_module.handle_gpu_compatibility = (
            lambda: call_log.__setitem__("gpu", call_log["gpu"] + 1)
        )
        io_utils_module.save_results_as_json = lambda main_path, results: call_log.__setitem__(
            "saved",
            (main_path, results),
        )
        model_utils_module.get_model = lambda input_shape, version: fake_model
        utils_package.common = common_module
        utils_package.data_utils = data_utils_module
        utils_package.eval_utils = eval_utils_module
        utils_package.io_utils = io_utils_module
        utils_package.model_utils = model_utils_module
        modules["utils"] = utils_package
        modules["utils.common"] = common_module
        modules["utils.data_utils"] = data_utils_module
        modules["utils.eval_utils"] = eval_utils_module
        modules["utils.io_utils"] = io_utils_module
        modules["utils.model_utils"] = model_utils_module

        trainer = import_project_module(
            "trainer",
            extra_modules=modules,
            clear_modules=("trainer",),
        )

        with redirect_stdout(io.StringIO()):
            trainer.main(["--version", "v1"])

        self.assertEqual(call_log["gpu"], 1)
        self.assertEqual(len(fake_model.add_calls), 1)
        self.assertEqual(fake_model.add_calls[0]["layer"], "Dense")
        self.assertEqual(fake_model.compile_calls[0]["loss"], "categorical_crossentropy")
        self.assertEqual(fake_model.compile_calls[0]["optimizer"]["learning_rate"], 0.1)
        self.assertEqual(fake_model.fit_calls[0][1]["batch_size"], 16)
        self.assertEqual(call_log["print_results"][0][1:], ("MNIST", "v1"))
        self.assertEqual(
            call_log["saved"],
            ("models/v1", {"MNIST": {"MNIST": {"loss": 0.1, "accuracy": 0.9}}}),
        )


if __name__ == "__main__":
    unittest.main()
