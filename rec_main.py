import argparse
import os
import time
from core.evaluate import evaluate_report
from core.preprocess import group_and_split
from core.prebuild import extract_services, construct_model
from core.predict import parallel_predict, predict_exec
from utils import syslog
from utils.helper import lyaml

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", type=str, required=True)
    argparser.add_argument("-o", type=str, required=True)
    argparser.add_argument("-c", type=str, required=True)
    argparser.add_argument("-s", type=str, required=True)
    args = argparser.parse_args()

    raw_input_dir = args.i
    config = lyaml(args.c)
    seeds_path = args.s
    output_dir = args.o

    save_input_dir = os.path.join(output_dir, config["save_dirname"]["input"])
    os.makedirs(save_input_dir, exist_ok=True)
    save_model_dir = os.path.join(output_dir, config["save_dirname"]["model"])
    os.makedirs(save_model_dir, exist_ok=True)
    save_output_dir = os.path.join(output_dir, config["save_dirname"]["output"])
    os.makedirs(save_output_dir, exist_ok=True)

    # time counting
    st_time = time.time()

    # data prepare
    group_and_split(
        raw_input_dir,
        seeds_path,
        save_input_dir,
        config["granuls"],
        config["chunksize"],
    )

    extract_services(save_input_dir, save_model_dir)
    construct_model(save_input_dir, save_model_dir)
    predict_time = parallel_predict(
        save_model_dir,
        save_input_dir,
        save_output_dir,
        config["budget"],
        config["num_workers"],
    )

    # end time counting
    ed_time = time.time()
    syslog.info(f"Total time cost: {ed_time - st_time}s")

    with open(os.path.join(output_dir, "time_report"), "w") as writer:
        writer.write(f"Predict time: {predict_time}s\n")
        # writer.write(f"Total time: {ed_time - st_time}s\n")

    save_eval_out_dir = os.path.join(output_dir, "eval")
    os.makedirs(save_eval_out_dir, exist_ok=True)
    evaluate_report(save_output_dir, save_eval_out_dir, config["budget"])
