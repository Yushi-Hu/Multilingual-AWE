import importlib
import functools


def partial_fun(path, **kwargs):
    mod, fun = path.rsplit(".", 1)
    fun = getattr(importlib.import_module(mod), fun)
    return functools.partial(fun, **kwargs)
