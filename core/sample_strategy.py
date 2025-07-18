from collections import Counter
from math import ceil
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from utils.str2idx import String2Index
from utils.helper import ljson
from utils import syslog

def percentage_sample(input_dir: str, output_path: str, granul: str, ratio: float):
    random.seed(2024)
    dfs = []
    for filename in tqdm(os.listdir(input_dir), desc="Loading"):
        dfs.append(
            pd.read_csv(os.path.join(input_dir, filename), usecols=["address", granul])
        )
    df = pd.concat(dfs)
    seeds = []
    bar = tqdm(total=len(df), desc="Percentage Sampling")
    for _, group in df.groupby(granul):
        seeds.extend(
            random.sample(group["address"].tolist(), k=ceil(len(group) * ratio))
        )
        bar.update(len(group))

    bar.close()

    syslog.info(f"Seeds: {len(seeds)} Total: {len(df)}")

    pd.DataFrame({"address": seeds}).to_csv(output_path, index=False)


def num_sample(
    input_dir: str, output_path: str, granul: str, ratio: float, num_seeds: int
):
    random.seed(2024)
    dfs = []
    for filename in tqdm(os.listdir(input_dir), desc="Loading"):
        dfs.append(
            pd.read_csv(os.path.join(input_dir, filename), usecols=["address", granul])
        )
    df = pd.concat(dfs)

    seeds = []
    bar = tqdm(total=len(df), desc="Percentage Sampling")
    for _, group in df.groupby(granul):
        sampleds = random.sample(
            group["address"].tolist(), k=ceil(len(group) * ratio)
        )
        seeds.extend(sampleds)
        if len(seeds) >= num_seeds:
            syslog.info(f"Reach the bound: {num_seeds}, actual: {len(seeds)}")
            break
        bar.update(len(group))

    bar.close()

    syslog.info(f"Seeds: {len(seeds)} Total: {len(df)}")

    pd.DataFrame({"address": seeds}).to_csv(output_path, index=False)

@dataclass
class SampleInstance:
    gvals: list[int]
    seqs: set[int]
    seq2addrs: dict[int, list[str]]

    def add_seq(self, seq: int):
        self.seqs.add(seq)

    def add_addr(self, seq: int, addr: str):
        if seq not in self.seq2addrs:
            self.seq2addrs[seq] = list[str]()
        self.seq2addrs[seq].append(addr)

    def limit(self, remove_seqs: set[int]):
        self.seqs.difference_update(remove_seqs)
        self.seq2addrs = {
            seq: self.seq2addrs[seq] for seq in self.seqs if seq in self.seqs
        }

    # def __repr__(self) -> str:
    #     return f"<SampleInstance gvals:{self.gvals} seqs: {len(self.seqs)} seq2addrs: {len(self.seq2addrs)}>"


@dataclass
class MetaInstance:
    subgval2seqs: dict[int, set[int]]
    seq2addrs: dict[int, list[str]]


def multigranul_sample(
    input_dir: str, output_path: str, granuls: list[str], max_ratio: float
):
    syslog.info(f"processing {input_dir}")
    random.seed(2024)

    port_flag = np.zeros(65536, dtype=np.bool)
    visited_seqstrs = set[str]()
    seqs = list[list[int]]()
    for filename in tqdm(os.listdir(input_dir), desc="Loading", dynamic_ncols=True):
        df = pd.read_csv(
            os.path.join(input_dir, filename),
            usecols=["port_service"],
        )

        for pses in tqdm(
            df["port_service"],
            total=len(df),
            desc=f"Preselect {filename}",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            seq = sorted(map(lambda pair: int(pair.split("#")[0]), ljson(pses)))
            seqstr = ",".join(map(str, seq))
            if seqstr in visited_seqstrs:
                continue
            visited_seqstrs.add(seqstr)
            port_flag[seq] = True
            seqs.append(seq)
    seqs.sort(key=lambda item: len(item), reverse=True)

    sel_seqs = list[list[int]]()
    for seq in tqdm(seqs, desc="Merging", dynamic_ncols=True):
        if np.any(port_flag[seq]):
            sel_seqs.append(seq)
            port_flag[seq] = False
        if not np.any(port_flag):
            syslog.info("Terminate by having covered all ports")
            break

    sel_seqstrs = set[str]()
    for seq in sel_seqs:
        sel_seqstrs.add(",".join(map(str, sorted(seq))))

    # 0 represents global, auto increment from 1
    # string to index
    gval2idx = String2Index()
    seqstr2idx = String2Index()

    # gval to instance
    gval2ins = dict[int, SampleInstance]()

    total_num_addrs = 0
    for filename in tqdm(os.listdir(input_dir), desc="Loading", dynamic_ncols=True):
        df = pd.read_csv(
            os.path.join(input_dir, filename),
            usecols=["address", "port_service"] + granuls,
        )
        total_num_addrs += len(df)
        # store and grouping
        for addr, pses, gvals in tqdm(
            zip(
                df["address"],
                df["port_service"],
                zip(*[df[granul] for granul in granuls]),
            ),
            total=len(df),
            desc=f"Grouping {filename}",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            # add global
            gvals = [0] + [gval2idx.get_index(gval) for gval in gvals]
            last_gval = gvals[-1]
            if last_gval not in gval2ins:
                gval2ins[last_gval] = SampleInstance(gvals, set(), {})
            ins = gval2ins[last_gval]
            seqstr = ",".join(
                map(
                    str,
                    sorted(map(lambda pair: int(pair.split("#")[0]), ljson(pses))),
                )
            )
            if seqstr not in sel_seqstrs:
                continue
            seq = seqstr2idx.get_index(seqstr)
            ins.add_seq(seq)
            ins.add_addr(seq, addr)

    syslog.info(f"total number of granul values {gval2idx._idx - 1}")
    syslog.info(f"total number of seqs {seqstr2idx._idx - 1}")

    is_done = False
    max_num_seeds = ceil(max_ratio * total_num_addrs)
    seeds = list[str]()
    instances = list(gval2ins.values())
    prev_gval2seqs = dict[int, set[int]]()

    # Start from 0th -> (n - 1)th
    curr_gidx = 0
    granuls = ["global"] + granuls
    while curr_gidx < len(granuls):
        prev_gidx = curr_gidx - 1
        next_gidx = curr_gidx + 1
        curr_gval2meta = dict[int, MetaInstance]()

        # compute the union seqs
        for ins in tqdm(
            instances,
            desc=f"Constructing {granuls[curr_gidx]} layer",
            dynamic_ncols=True,
        ):

            # previous sequences
            prev_seqs = set[int]()
            if prev_gidx != -1:
                prev_seqs = prev_gval2seqs[ins.gvals[prev_gidx]]
            ins.limit(prev_seqs)

            # current sequences, remove previouse sequences
            curr_gval = ins.gvals[curr_gidx]
            if next_gidx >= len(granuls):
                next_gval = -1
            else:
                next_gval = ins.gvals[next_gidx]
            if curr_gval not in curr_gval2meta:
                curr_gval2meta[curr_gval] = MetaInstance(
                    {next_gval: ins.seqs},
                    {seq: addrs.copy() for seq, addrs in ins.seq2addrs.items()},
                )
            else:
                assert next_gval != -1, f"Err: duplicate BGP prefixes?"
                meta = curr_gval2meta[curr_gval]
                if next_gval not in meta.subgval2seqs:
                    meta.subgval2seqs[next_gval] = ins.seqs
                else:
                    meta.subgval2seqs[next_gval] = (
                        ins.seqs | meta.subgval2seqs[next_gval]
                    )
                for seq, addrs in ins.seq2addrs.items():
                    if seq not in meta.seq2addrs:
                        meta.seq2addrs[seq] = addrs.copy()
                    else:
                        meta.seq2addrs[seq].extend(addrs)

        prev_gval2seqs.clear()
        debug_sampled_seqs = 0
        for gval, meta in tqdm(
            curr_gval2meta.items(),
            desc=f"Sampling {granuls[curr_gidx]} layer",
            dynamic_ncols=True,
        ):
            counter = Counter[int]()
            for seqs in meta.subgval2seqs.values():
                counter.update(seqs)
            sel_seqs = set[int]()
            for seq, count in counter.most_common():
                if count > 1 or curr_gidx == len(granuls) - 1:
                    sel_seqs.add(seq)
            prev_gval2seqs[gval] = sel_seqs
            for seq, addrs in meta.seq2addrs.items():
                if seq not in sel_seqs:
                    continue
                seeds.append(random.choice(addrs))
                if len(seeds) >= max_num_seeds:
                    is_done = True
                    break
                debug_sampled_seqs += 1
            if is_done:
                syslog.info("Terminated by reaching the maximum sampling number")
                break
        syslog.debug(f"{granuls[curr_gidx]} Sampled seqs:  {debug_sampled_seqs}")
        if is_done:
            break
        curr_gidx += 1

    syslog.info(
        f"Seeds: {len(seeds)} Total: {total_num_addrs} R: {len(seeds) / total_num_addrs}"
    )
    pd.DataFrame({"address": seeds}).to_csv(output_path, index=False)
