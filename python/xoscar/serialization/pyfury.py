import os
import threading

_fury = threading.local()


def get_fury():
    if os.environ.get("USE_FURY") in ("1", "true", "True"):
        instance = getattr(_fury, "instance", None)
        if instance is not None:
            return instance
        else:
            try:
                import pyfury

                _fury.instance = instance = pyfury.Fury(
                    language=pyfury.Language.PYTHON, require_class_registration=False
                )
                print("pyfury is enabled.")
            except ImportError:
                print("pyfury is not installed.")
            return instance
