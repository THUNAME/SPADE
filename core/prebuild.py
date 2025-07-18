import os
import pandas as pd
from tqdm import tqdm
import typing
from utils.helper import ljson, fsjson
from .models import GraphSeer, FreqSeer

SERVICES_FILENAME = "services.json"
GLOBAL_MODEL_NAME = "global"


def extract_services(input_dir: str, model_dir: str):
    svcs = set[str]()
    for filename in tqdm(
        os.listdir(input_dir),
        desc="Extracting services",
        dynamic_ncols=True,
    ):
        df = pd.read_csv(
            os.path.join(input_dir, filename),
            usecols=["port_service", "is_seed"],
            na_filter=False,
        ).query("is_seed==1")
        for pses in df["port_service"]:
            for _, svc in ljson(pses):
                svcs.add(svc)
    fsjson(os.path.join(model_dir, SERVICES_FILENAME), sorted(svcs))


class RecursiveBuilder:
    def __init__(self, num_models: int, model_dir: str, svcs_path: str) -> None:
        self._bar = tqdm(total=num_models, desc="Building models", dynamic_ncols=True)
        self._model_dir = model_dir
        self._svcs_path = svcs_path

    def build(
        self,
        gval2data: typing.Union[dict, list[list[tuple[int, str]]]],
        curr_gval: str,
    ):
        fseer = FreqSeer()
        gseer = GraphSeer(self._svcs_path)

        if type(gval2data) == list:
            # do not need to build a model
            return gval2data
        elif type(gval2data) == dict:
            # need to build offline models
            batch_pses = list[list[tuple[int, str]]]()
            for gval, data in gval2data.items():
                batch_pses.extend(self.build(data, str(gval)))
                self._bar.update()
        else:
            raise Exception(f"Err: Unknown type found {type(gval2data)}")

        fseer.update(batch_pses)
        gseer.update(batch_pses)

        fseer.save(self._model_dir, curr_gval)
        gseer.save(self._model_dir, curr_gval)
        return batch_pses

    def __del__(self):
        self._bar.close()


def construct_model(input_dir: str, model_dir: str):
    svcs_path = os.path.join(model_dir, SERVICES_FILENAME)

    # grouping
    num_models = 0
    gval2data = dict()
    for filename in tqdm(
        os.listdir(input_dir), desc="Grouping seeds", dynamic_ncols=True
    ):
        df = pd.read_csv(
            os.path.join(input_dir, filename),
            usecols=["port_service", "is_seed", "gvals"],
            na_filter=False,
        ).query("is_seed==1")
        for pses, gvals in zip(df["port_service"], df["gvals"]):
            pses = ljson(pses)
            gvals = ljson(gvals)

            ptr = gval2data
            for gval in gvals[:-1]:
                if gval not in ptr:
                    num_models += 1
                    ptr[gval] = {}
                ptr = ptr[gval]
            last_gval = gvals[-1]

            if last_gval not in ptr:
                num_models += 1
                ptr[last_gval] = []
            ptr[last_gval].append(pses)

    RecursiveBuilder(num_models, model_dir, svcs_path).build(gval2data, GLOBAL_MODEL_NAME)
