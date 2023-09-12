import os
import threading

_fury = threading.local()
_fury_not_installed = object()
_register_class_list = set()


def register_classes(*args):
    instance = get_fury()
    if instance is not None:
        _register_class_list.update(args)
        for c in _register_class_list:
            instance.register_class(c)


def get_fury():
    if os.environ.get("USE_FURY") in ("1", "true", "True"):
        instance = getattr(_fury, "instance", None)
        if instance is _fury_not_installed:  # pragma: no cover
            return None
        if instance is not None:
            return instance
        else:
            try:
                import pyfury

                _fury.instance = instance = pyfury.Fury(
                    language=pyfury.Language.PYTHON, require_class_registration=False
                )
                for c in _register_class_list:  # pragma: no cover
                    instance.register_class(c)
                print("pyfury is enabled.")
            except ImportError:  # pragma: no cover
                print("pyfury is not installed.")
                _fury.instance = _fury_not_installed
            return instance
