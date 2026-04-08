"""Helpers for importing project modules without third-party ML dependencies."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Dict, Mapping, Optional, Sequence, Tuple


ModuleMap = Dict[str, types.ModuleType]


class FakeFigure:
    """Minimal figure object used by the matplotlib test stub."""

    def colorbar(self, *_args: object, **_kwargs: object) -> None:
        """Accept colorbar calls without doing any work."""

    def tight_layout(self) -> None:
        """Accept layout calls without doing any work."""


class FakeAxes:
    """Minimal axes object used by the matplotlib test stub."""

    def __init__(self) -> None:
        """Initialize the axes stub."""
        self.figure = FakeFigure()
        self.last_set = {}

    def imshow(self, *_args: object, **_kwargs: object) -> object:
        """Accept image rendering calls."""
        return object()

    def set(self, **kwargs: object) -> None:
        """Record the latest axes configuration."""
        self.last_set = kwargs

    def get_xticklabels(self) -> Sequence[object]:
        """Return an empty label list for pyplot.setp."""
        return []

    def text(self, *_args: object, **_kwargs: object) -> None:
        """Accept text rendering calls without doing any work."""


class FakeImage:
    """Small image object used by the PIL test stub."""

    def __init__(self, array: object) -> None:
        """Store the original array for traceability in tests."""
        self.array = array

    def resize(self, _size: Tuple[int, int]) -> "FakeImage":
        """Return the image object so resize calls can be chained."""
        return self

    def convert(self, _mode: str) -> "FakeImage":
        """Return the image object so conversion calls can be chained."""
        return self


class FakeSequential:
    """Minimal Sequential replacement that records added layers."""

    def __init__(self) -> None:
        """Initialize an empty layer list."""
        self.layers = []

    def add(self, layer: Mapping[str, object]) -> None:
        """Append a layer configuration to the internal list."""
        self.layers.append(layer)


class TensorFlowStub(types.ModuleType):
    """Minimal TensorFlow stub with enough Keras surface for unit tests."""

    def __init__(
        self,
        gpus: Optional[Sequence[str]] = None,
        image_data_format: str = "channels_last",
    ) -> None:
        """Initialize the TensorFlow stub and its nested Keras modules."""
        super().__init__("tensorflow")
        self.memory_growth_calls = []
        experimental = types.SimpleNamespace(
            list_physical_devices=lambda _device_type: list(gpus or []),
            set_memory_growth=self._set_memory_growth,
        )
        self.config = types.SimpleNamespace(experimental=experimental)

        keras_module = types.ModuleType("tensorflow.keras")
        backend_module = types.ModuleType("tensorflow.keras.backend")
        backend_module.image_data_format = lambda: image_data_format

        layers_module = types.ModuleType("tensorflow.keras.layers")
        for layer_name in (
            "BatchNormalization",
            "Conv2D",
            "Dense",
            "Dropout",
            "Flatten",
            "MaxPooling2D",
        ):
            setattr(layers_module, layer_name, make_layer_factory(layer_name))

        callbacks_module = types.ModuleType("tensorflow.keras.callbacks")
        callbacks_module.EarlyStopping = make_layer_factory("EarlyStopping")
        callbacks_module.ModelCheckpoint = make_layer_factory("ModelCheckpoint")

        models_module = types.ModuleType("tensorflow.keras.models")
        models_module.Sequential = FakeSequential

        datasets_module = types.ModuleType("tensorflow.keras.datasets")
        datasets_module.mnist = types.SimpleNamespace(load_data=lambda: (([], []), ([], [])))

        keras_module.backend = backend_module
        keras_module.callbacks = callbacks_module
        keras_module.datasets = datasets_module
        keras_module.layers = layers_module
        keras_module.losses = types.SimpleNamespace(categorical_crossentropy="categorical_crossentropy")
        keras_module.models = models_module
        keras_module.optimizers = types.SimpleNamespace(
            Adadelta=lambda learning_rate: {
                "optimizer": "Adadelta",
                "learning_rate": learning_rate,
            }
        )
        keras_module.utils = types.SimpleNamespace(
            to_categorical=lambda output, num_classes: {
                "encoded": output,
                "num_classes": num_classes,
            }
        )

        self.keras = keras_module
        self.modules = {
            "tensorflow": self,
            "tensorflow.keras": keras_module,
            "tensorflow.keras.backend": backend_module,
            "tensorflow.keras.callbacks": callbacks_module,
            "tensorflow.keras.datasets": datasets_module,
            "tensorflow.keras.layers": layers_module,
            "tensorflow.keras.models": models_module,
        }

    def _set_memory_growth(self, gpu: str, enabled: bool) -> None:
        """Record GPU memory-growth requests made by the code under test."""
        self.memory_growth_calls.append((gpu, enabled))


def make_layer_factory(layer_name: str) -> types.FunctionType:
    """Create a callable that records layer construction metadata.

    Args:
        layer_name (str): Name of the emulated Keras object.

    Returns:
        types.FunctionType: Callable layer factory.
    """

    def factory(*args: object, **kwargs: object) -> Mapping[str, object]:
        """Return the recorded layer metadata."""
        return {
            "layer": layer_name,
            "args": args,
            "kwargs": kwargs,
        }

    return factory


def make_dependency_modules(
    gpus: Optional[Sequence[str]] = None,
    image_data_format: str = "channels_last",
) -> Tuple[TensorFlowStub, ModuleMap]:
    """Build stub modules required to import project files in bare Python.

    Args:
        gpus (Optional[Sequence[str]]): GPU names visible through the TF stub.
        image_data_format (str): Value returned by the Keras backend stub.

    Returns:
        Tuple[TensorFlowStub, ModuleMap]: TensorFlow stub and the module map.
    """
    tensorflow_stub = TensorFlowStub(gpus=gpus, image_data_format=image_data_format)

    numpy_module = types.ModuleType("numpy")
    numpy_module.ndarray = object
    numpy_module.newaxis = object()
    numpy_module.arange = lambda value: list(range(value))
    numpy_module.array = lambda value: value
    numpy_module.asarray = lambda value: value
    numpy_module.concatenate = lambda values: sum((list(value) for value in values), [])
    numpy_module.divide = lambda numerator, denominator, out=None, where=None: numerator
    numpy_module.loadtxt = lambda *args, **kwargs: []
    numpy_module.zeros_like = lambda value: value

    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    pyplot_module.cm = types.SimpleNamespace(Blues="Blues")
    pyplot_module.subplots = lambda: (FakeFigure(), FakeAxes())
    pyplot_module.savefig = lambda *_args, **_kwargs: None
    pyplot_module.setp = lambda *_args, **_kwargs: None
    pyplot_module.margins = lambda *_args, **_kwargs: None
    matplotlib_module.pyplot = pyplot_module

    pil_module = types.ModuleType("PIL")
    pil_module.Image = types.SimpleNamespace(fromarray=lambda array: FakeImage(array))

    scipy_module = types.ModuleType("scipy")
    scipy_io_module = types.ModuleType("scipy.io")
    scipy_io_module.loadmat = lambda _path: {}
    scipy_module.io = scipy_io_module

    sklearn_module = types.ModuleType("sklearn")
    sklearn_datasets_module = types.ModuleType("sklearn.datasets")
    sklearn_datasets_module.load_digits = lambda: types.SimpleNamespace(images=[], target=[])
    sklearn_metrics_module = types.ModuleType("sklearn.metrics")
    sklearn_metrics_module.accuracy_score = lambda *_args, **_kwargs: 0.0
    sklearn_metrics_module.classification_report = lambda *_args, **_kwargs: "report"
    sklearn_metrics_module.confusion_matrix = lambda *_args, **_kwargs: [[1]]
    sklearn_model_selection_module = types.ModuleType("sklearn.model_selection")
    sklearn_model_selection_module.train_test_split = (
        lambda x, y, test_size=0.2: (x, x, y, y)
    )
    sklearn_module.datasets = sklearn_datasets_module
    sklearn_module.metrics = sklearn_metrics_module
    sklearn_module.model_selection = sklearn_model_selection_module

    modules = {
        "PIL": pil_module,
        "matplotlib": matplotlib_module,
        "matplotlib.pyplot": pyplot_module,
        "numpy": numpy_module,
        "scipy": scipy_module,
        "scipy.io": scipy_io_module,
        "sklearn": sklearn_module,
        "sklearn.datasets": sklearn_datasets_module,
        "sklearn.metrics": sklearn_metrics_module,
        "sklearn.model_selection": sklearn_model_selection_module,
    }
    modules.update(tensorflow_stub.modules)

    return tensorflow_stub, modules


def import_project_module(
    module_name: str,
    extra_modules: Optional[Mapping[str, types.ModuleType]] = None,
    clear_modules: Optional[Sequence[str]] = None,
) -> types.ModuleType:
    """Import a project module while temporarily injecting dependency stubs.

    Args:
        module_name (str): Fully qualified module name to import.
        extra_modules (Optional[Mapping[str, types.ModuleType]]): Modules inserted
            into `sys.modules` during import.
        clear_modules (Optional[Sequence[str]]): Modules removed from
            `sys.modules` before importing.

    Returns:
        types.ModuleType: Imported module.
    """
    extra_modules = dict(extra_modules or {})
    clear_modules = tuple(clear_modules or ())

    original_modules = {}
    for name, module in extra_modules.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    for name in (module_name,) + clear_modules:
        sys.modules.pop(name, None)

    try:
        imported_module = importlib.import_module(module_name)
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module

    return imported_module
