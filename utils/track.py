import time
from typing import Callable, Any
import inspect


def timer(fn: Callable[..., Any]):
    def wrapper(*args, **kwargs):
        st_time = time.time()
        rt = fn(*args, **kwargs)
        ed_time = time.time()
        return rt, ed_time - st_time

    return wrapper


def time_track(fn: Callable[..., Any]):
    def wrapper(*args, **kwargs):
        tick = time.time()
        rt = fn(*args, **kwargs)
        print(f"{inspect.getfile(fn)}.{fn.__name__} time cost: {time.time() - tick}")
        return rt

    return wrapper


class TimeTrack:
    def __init__(self, desc=""):
        self._tick = 0
        self._desc = desc

    def __enter__(self):
        self._tick = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"{self._desc}time cost: {time.time() - self._tick}")
