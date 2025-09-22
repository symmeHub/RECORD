"""
Small collection of decorators used across the core package.
"""

import threading
from functools import wraps


def threaded(fn):
    """
    Run the decorated function in a daemon thread.

    Returns the started `threading.Thread`.
    """

    def wrapper(*args, **kwargs):
        th = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        th.start()
        return th

    return wrapper


def execute_once(func):
    """
    Ensure the decorated function runs at most once.

    - Subsequent calls return None by default.
    - A `reset()` method is attached to the wrapper to allow re-running once again.
    """
    has_run = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)
        return None  # Optionally return something when the function is skipped

    def reset():
        nonlocal has_run
        has_run = False

    wrapper.reset = reset
    return wrapper
