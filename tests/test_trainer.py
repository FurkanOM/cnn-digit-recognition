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
        helper_module = types.ModuleType("helpers")
        helper_module.calls = []

        def unexpected_call(*_args: object, **_kwargs: object) -> None:
            """Fail if import-time side effects trigger helper work."""
            helper_module.calls.append("called")

        helper_module.handle_args = unexpected_call
        helper_module.handle_gpu_compatibility = unexpected_call
        helper_module.get_datasets = unexpected_call
        helper_module.get_input_shape = unexpected_call
        helper_module.get_main_path = unexpected_call
        helper_module.get_model = unexpected_call
        helper_module.get_model_path = unexpected_call
        helper_module.get_batch_size = unexpected_call
        helper_module.evaluate = unexpected_call
        helper_module.print_results = unexpected_call
        helper_module.save_results_as_json = unexpected_call
        modules["helpers"] = helper_module

        import_project_module(
            "trainer",
            extra_modules=modules,
            clear_modules=("trainer",),
        )

        self.assertEqual(helper_module.calls, [])

    def test_main_runs_training_flow_and_persists_results(self) -> None:
        """The trainer should build, fit, evaluate, and persist each dataset run."""
        _, modules = make_dependency_modules()
        helper_module = types.ModuleType("helpers")
        fake_model = FakeTrainingModel()
        call_log = {
            "gpu": 0,
            "print_results": [],
            "saved": None,
        }
        datasets = {
            "MNIST": {
                "x_train": types.SimpleNamespace(shape=(10,)),
                "y_train": [1],
                "x_valid": [2],
                "y_valid": [3],
            }
        }

        helper_module.handle_args = lambda argv=None: types.SimpleNamespace(handle_gpu=True, version="v1")
        helper_module.handle_gpu_compatibility = lambda: call_log.__setitem__("gpu", call_log["gpu"] + 1)
        helper_module.get_datasets = lambda use_datasets, n_combinations=1: datasets
        helper_module.get_input_shape = lambda: (28, 28, 1)
        helper_module.get_main_path = lambda version: "models/" + version
        helper_module.get_model = lambda input_shape, version: fake_model
        helper_module.get_model_path = lambda main_path, trained_with: main_path + "/" + trained_with + ".h5"
        helper_module.get_batch_size = lambda x_train: 16
        helper_module.evaluate = lambda model, dataset_map, use_datasets: {
            "MNIST": {"loss": 0.1, "accuracy": 0.9}
        }
        helper_module.print_results = (
            lambda results, trained_with, version: call_log["print_results"].append(
                (results, trained_with, version)
            )
        )
        helper_module.save_results_as_json = lambda main_path, results: call_log.__setitem__(
            "saved",
            (main_path, results),
        )
        modules["helpers"] = helper_module

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
