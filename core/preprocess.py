import os
import pandas as pd
from tqdm import tqdm
from utils import syslog
from utils.helper import ljson, tjson
from utils.str2idx import String2Index


def group_and_split(
    raw_dir: str,
    seeds_path: str,
    out_dir: str,
    granuls: list[str],
    chunksize: int,
):
    os.makedirs(out_dir, exist_ok=True)

    # loading the raw data
    df = pd.concat(
        [
            pd.read_csv(
                os.path.join(raw_dir, filename),
                usecols=["address", "port_service"] + granuls,
                na_filter=False
            )
            for filename in tqdm(
                os.listdir(raw_dir),
                desc=f"Loading from {raw_dir}",
                dynamic_ncols=True,
            )
        ]
    )

    def to_pair(port_service: str):
        port, service = port_service.split("#")
        port = int(port)
        return [port, service]

    # transform granul value to index
    gval2idx = String2Index()

    # loading seeds
    seeds_set = set(pd.read_csv(seeds_path, na_filter=False)["address"])

    syslog.info(f"Using {len(seeds_set)} seeds")

    # grouping
    group2data = dict[int, list]()
    for addr, port_svcs, gvals in tqdm(
        zip(
            df["address"], df["port_service"], zip(*[df[granul] for granul in granuls])
        ),
        total=len(df),
        desc="Making Groups",
        dynamic_ncols=True,
    ):
        gvals = [gval2idx.get_index(gval) for gval in gvals]
        last_gval = gvals[-1]
        if last_gval not in group2data:
            group2data[last_gval] = []
        group2data[last_gval].append(
            {
                "address": addr,
                "gvals": tjson(gvals),
                "port_service": tjson(list(map(to_pair, ljson(port_svcs)))),
                "is_seed": int(addr in seeds_set),
            }
        )

    index = 0
    chunkdata = []
    processed_size = 0

    with tqdm(
        total=len(df), desc=f"Grouping and spliting {granuls}", dynamic_ncols=True
    ) as bar:
        for _, data in group2data.items():
            chunkdata.extend(data)
            if len(chunkdata) >= chunksize:
                pd.DataFrame(chunkdata).to_csv(
                    os.path.join(out_dir, f"chunk_{index}.csv"), index=False
                )
                processed_size += len(chunkdata)
                syslog.info(
                    f"Save chunk {index:2} ({len(chunkdata)} addresses). Processed {processed_size / len(df) * 100: .2f}%"
                )
                index += 1
                chunkdata.clear()
            bar.update(len(data))

    if len(chunkdata) > 0:
        pd.DataFrame(chunkdata).to_csv(
            os.path.join(out_dir, f"chunk_{index}.csv"), index=False
        )
        processed_size += len(chunkdata)
        syslog.info(
            f"Save chunk {index:2} ({len(chunkdata)} addresses). Processed {processed_size / len(df) * 100: .2f}%"
        )
        index += 1
        chunkdata.clear()
