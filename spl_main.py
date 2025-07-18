import argparse
from core.sample_strategy import multigranul_sample
from utils.helper import lyaml

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", type=str, required=True)
    argparser.add_argument("-o", type=str, required=True)
    argparser.add_argument("-c", type=str, required=True)
    argparser.add_argument("--r", type=float, default=0.1)
    args = argparser.parse_args()
    in_dir = args.i
    output_path = args.o
    config = lyaml(args.c)
    max_ratio = args.p
    multigranul_sample(
        in_dir,
        output_path,
        config["granuls"],
        max_ratio,
    )
