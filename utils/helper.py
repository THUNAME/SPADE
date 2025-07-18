import os
import yaml
import json
import pickle
import random
import numpy as np
from typing import Any


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)


def fsjson(file: str, obj: Any):
    r"""
    save json file
    """
    with open(file, "w", encoding="utf8") as fw:
        json.dump(obj, fw)


def fljson(file: str):
    r"""
    load json file
    """
    with open(file, "r", encoding="utf8") as fr:
        return json.load(fr)


def fspkl(file: str, obj: Any):
    r"""
    save pickle file
    """
    with open(file, "wb") as fw:
        pickle.dump(obj, fw)


def flpkl(file: str):
    r"""
    load pickle file
    """
    with open(file, "rb") as fr:
        return pickle.load(fr)


def tpkl(obj: Any):
    r"""
    [pickle] to bytes
    """
    return pickle.dumps(obj)


def lpkl(obj: bytes):
    r"""
    [pickle] load obj
    """
    return pickle.loads(obj)


def tjson(obj: Any):
    r"""
    [json] to string
    """
    return json.dumps(obj)


def ljson(obj: str):
    r"""
    [json] load obj
    """
    return json.loads(obj)


def lyaml(filepath: str):
    with open(filepath, "rb") as fr:
        obj = yaml.load(fr, yaml.SafeLoader)
    return obj


def cp_dir(src: str, dst: str):
    if os.system(f"cp -rf {src} {dst}") != 0:
        raise Exception(f"Fail to cp {src} to {dst}")
