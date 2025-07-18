import os
import numpy as np
import scipy.sparse as sp
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Union
from utils.helper import fsjson, fljson


class FreqSeer:
    def __init__(self) -> None:
        self._port2svc_ctr = dict[int, Counter[str]]()
        self._dft = Counter[str]()

    def predict(self, port: int):
        return map(
            lambda pair: pair[0], self._port2svc_ctr.get(port, self._dft).most_common()
        )

    def update(self, batch_pses: list[list[tuple[int, str]]]):
        for pses in batch_pses:
            for port, svc in pses:
                if port not in self._port2svc_ctr:
                    self._port2svc_ctr[port] = Counter()
                self._port2svc_ctr[port].update([svc])

    def save(self, save_dir: str, name: str):
        fsjson(
            os.path.join(save_dir, f"{name}.json"),
            [
                [port, {svc: cnt for svc, cnt in svc_ctr.most_common()}]
                for port, svc_ctr in self._port2svc_ctr.items()
            ],
        )

    @staticmethod
    def save_path(save_dir: str, name: str):
        return os.path.join(save_dir, f"{name}.json")

    @staticmethod
    def from_files(load_dir: str, name: str):
        fseer = FreqSeer()
        for port, svc_freq in fljson(os.path.join(load_dir, f"{name}.json")):
            ctr = Counter[str]()
            for svc, cnt in svc_freq.items():
                ctr[svc] = cnt
            fseer._port2svc_ctr[port] = ctr
        return fseer


class GraphSeer:
    def __init__(self, svcs: Union[str, list[str]]):
        r"""
        :param svcs: kinds of services (list or filepath)
        """
        if type(svcs) == str:
            self._svcs: list[str] = fljson(svcs)
        elif type(svcs) == list:
            self._svcs = svcs.copy()
        else:
            raise Exception(f"Unknown {type(svcs)}")
        self._s2i = dict[str, int]()
        self._i2s = dict[int, str]()
        for index, svc in enumerate(self._svcs):
            self._s2i[svc] = index
            self._i2s[index] = svc
        self._nsvc = len(self._svcs)
        # set node frequence
        self._nfreq = np.zeros((self._nsvc,), dtype=np.float32)
        # set adjacent matrix
        self._adj = np.zeros((self._nsvc, self._nsvc), dtype=np.float32)

    def predict(self, running_svcs: set[str]) -> Iterator[str]:
        freq_sum = self._nfreq.sum()
        if freq_sum == 0:
            return
        nproba = self._nfreq / freq_sum
        for svc in running_svcs:
            # do max
            idx = self._s2i[svc]
            freq = self._nfreq[idx]
            if freq == 0:
                continue
            np.maximum(nproba, self._adj[idx] / freq, out=nproba)

        while True:
            idx = nproba.argmax()
            if nproba[idx] == 0:
                # proba is 0, exit
                break
            nproba[idx] = 0
            yield self._i2s[idx]

    def update(self, batch_pses: list[list[tuple[int, str]]]):
        if len(batch_pses) == 0:
            return
        row = []
        col = []
        for pses in batch_pses:
            svcs = list[str]()
            for svc, cnt in Counter(map(lambda ps: ps[1], pses)).most_common():
                if cnt > 2:
                    # self-weight
                    svcs.append(svc)
                    svcs.append(svc)
                else:
                    # other-weight
                    svcs.append(svc)
            update_nids = []
            for i in range(len(svcs)):
                src = self._s2i[svcs[i]]
                update_nids.append(src)
                for j in range(i + 1, len(svcs)):
                    dst = self._s2i[svcs[j]]
                    # src -> dst
                    row.append(src)
                    col.append(dst)
                    # dst -> src
                    row.append(dst)
                    col.append(src)
            self._nfreq[update_nids] += 1
        self._adj += sp.csr_array(
            ([1] * len(row), (row, col)),
            shape=(self._nsvc, self._nsvc),
            dtype=np.float32,
        ).toarray()

    def save(self, save_dir: str, name: str):
        # nfreq adj
        np.savez(
            os.path.join(save_dir, f"{name}.npz"), nfreq=self._nfreq, adj=self._adj
        )

    @staticmethod
    def save_path(save_dir: str, name: str):
        return os.path.join(save_dir, f"{name}.npz")

    @staticmethod
    def from_files(load_dir: str, name: str, svcs: Union[str, list[str]]):
        gseer = GraphSeer(svcs)
        model = np.load(os.path.join(load_dir, f"{name}.npz"))
        gseer._nfreq = model["nfreq"]
        gseer._adj = model["adj"]
        return gseer


@dataclass
class ServiceSeer:
    fseer: FreqSeer
    gseer: GraphSeer

    def update(self, batch_pses: list[list[tuple[int, str]]]):
        self.fseer.update(batch_pses)
        self.gseer.update(batch_pses)

    @staticmethod
    def exist(load_dir: str, name: str):
        return os.path.exists(FreqSeer.save_path(load_dir, name)) and os.path.exists(
            GraphSeer.save_path(load_dir, name)
        )

    @staticmethod
    def new(svcs: Union[str, list[str]]):
        return ServiceSeer(FreqSeer(), GraphSeer(svcs))

    @staticmethod
    def from_files(load_dir: str, name: str, svcs: Union[str, list[str]]):
        return ServiceSeer(
            FreqSeer.from_files(load_dir, name),
            GraphSeer.from_files(load_dir, name, svcs),
        )
