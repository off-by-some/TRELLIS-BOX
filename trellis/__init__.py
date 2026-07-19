import importlib


__all__ = [
    "models",
    "modules",
    "pipelines",
    "renderers",
    "representations",
    "utils",
]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
