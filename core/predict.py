import os
import time
import pandas as pd
from tqdm import tqdm
from utils import syslog
from utils.helper import ljson, tjson
from .models import FreqSeer, GraphSeer, ServiceSeer
from .prebuild import SERVICES_FILENAME, GLOBAL_MODEL_NAME
from concurrent.futures import ProcessPoolExecutor, as_completed

GROUNDTRUTH = 0
IS_DONE = 1
PREDICTS = 2


def _freq_predict(
    fseer: FreqSeer,
    port2state: dict[int, list],
    budget: int,
):
    r"""
    :param fseer: port-service seer
    :param port2state: a dict of [true, is_done, predicts]
    :param budget: total predict budget
    """
    for port, state in port2state.items():
        groundtruth, _, predicts = state

        # NOTE: skip
        if state[IS_DONE]:
            continue

        # NOTE: load predicted services
        predicted = set(predicts)

        for svc in fseer.predict(port):
            if len(predicts) > budget:
                # out of budgets
                break
            if svc in predicted:
                # have been predicted
                continue
            # predict this one
            predicts.append(svc)
            # set is_done = true, finish predicting
            if svc == groundtruth:
                state[IS_DONE] = True
                break


def _graph_predict(
    gseer: GraphSeer,
    port2state: dict[int, list],
    budget: int,
):
    # running service of an address
    running_svcs = set[str]()

    # the rest of ports which have not been predicted
    rest_ports = list[int]()
    for port, (groundtruth, is_done, predicts) in port2state.items():
        if is_done:
            # is done
            running_svcs.add(groundtruth)
        elif len(predicts) < budget:
            # isn't done, and isn't fully being predicted
            rest_ports.append(port)

        # pass the rest

    # do predict until no ports left
    for port in rest_ports:
        state = port2state[port]
        groundtruth, _, predicts = state

        # load predicted services
        predicted = set(predicts)
        # do predict
        for svc in gseer.predict(running_svcs):
            if len(predicts) > budget:
                # out of budgets
                break
            if svc in predicted:
                # have been predicted
                continue
            # predict
            predicts.append(svc)
            # is true
            if svc == groundtruth:
                # set is done
                state[IS_DONE] = True

                # update running services
                running_svcs.add(svc)
                break


def multigranul_predict(
    sseers: list[ServiceSeer],
    svcs_path: str,
    df: pd.DataFrame,
    budget: int,
    outputs: list[dict],
    bar: tqdm,
):
    # split data
    ports = set[int]()
    train = list[list[tuple[int, str]]]()
    test = list[tuple[str, dict[int, list]]]()
    for gvals, port_svcs, is_seed in zip(
        df["gvals"], df["port_service"], df["is_seed"]
    ):
        port_svcs = ljson(port_svcs)
        if is_seed:
            train.append(port_svcs)
        else:
            test.append((gvals, {port: [svc, False, []] for port, svc in port_svcs}))
            # port -> [groundtruth, is_done, predicts]

        # count port
        for port, _ in port_svcs:
            ports.add(port)
    # create freq seer, local graph seer
    lsseer = ServiceSeer.new(svcs_path)
    lsseer.update(train)

    # predict
    for gvals, port2state in test:
        # freq predict
        _freq_predict(lsseer.fseer, port2state, budget)
        for sseer in sseers:
            # fine -> corse
            _freq_predict(sseer.fseer, port2state, budget)

        # graph predict
        _graph_predict(lsseer.gseer, port2state, budget)
        for sseer in sseers:
            # fine -> corse
            _graph_predict(sseer.gseer, port2state, budget)

        # update
        port_svcs = list[tuple[int, str]]()
        record = []
        for port, (groundtruth, is_done, predicts) in port2state.items():
            if is_done:
                port_svcs.append((port, groundtruth))
            record.append([port, groundtruth, predicts])

        # update
        lsseer.update([port_svcs])

        # save
        outputs.append({"gvals": gvals, "record": tjson(record)})
        bar.update()


def _loading_sseers(
    model_dir: str, svcs_path: str, idx2sseer: dict[str, ServiceSeer], gvals: list[int]
) -> list[ServiceSeer]:
    # add default global model
    gvalstrs = [GLOBAL_MODEL_NAME] + [str(gval) for gval in gvals]
    sseers = list[ServiceSeer]()
    # ignore last gvalstr
    for gvalstr in reversed(gvalstrs[:-1]):
        if gvalstr not in idx2sseer:
            # check json?
            if not ServiceSeer.exist(model_dir, gvalstr):
                continue
            idx2sseer[gvalstr] = ServiceSeer.from_files(model_dir, gvalstr, svcs_path)
        sseers.append(idx2sseer[gvalstr])
    return sseers


def predict_exec(
    model_dir: str,
    filepath: str,
    output_dir: str,
    budget: int,
):
    svcs_path = os.path.join(model_dir, SERVICES_FILENAME)
    idx2sseer = dict[str, ServiceSeer]()

    df = pd.read_csv(filepath, na_filter=False)
    outputs = list[dict]()
    with tqdm(
        total=len(df.query("is_seed == 0")),
        desc=f"<P-{os.getpid()}> PRED",
        dynamic_ncols=True,
    ) as bar:
        for gvals, g_df in df.groupby("gvals"):
            gvals = ljson(str(gvals))
            sseers = _loading_sseers(model_dir, svcs_path, idx2sseer, gvals)
            multigranul_predict(sseers, svcs_path, g_df, budget, outputs, bar)
    pd.DataFrame(outputs).to_csv(
        os.path.join(output_dir, os.path.basename(filepath)), index=False
    )


def parallel_predict(
    model_dir: str,
    input_dir: str,
    output_dir: str,
    budget: int,
    num_workers: int,
):
    input_files = os.listdir(input_dir)
    st_time = time.time()
    with ProcessPoolExecutor(min(len(input_files), num_workers)) as executor:
        done = 0
        tasks = [
            executor.submit(
                predict_exec,
                model_dir,
                os.path.join(input_dir, filename),
                output_dir,
                budget,
            )
            for filename in input_files
        ]
        for task in as_completed(tasks):
            done += 1
            syslog.info(f"DONE {done}/{len(tasks)}")
        syslog.info(f"DONE {len(tasks)} tasks")
    ed_time = time.time()
    syslog.info(f"Total testing time cost: {ed_time - st_time}s")
    return ed_time - st_time
