# predict true, predict wrong, can not predict, budget
# start from 1!!!
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.helper import ljson


def evaluate_report_chunk(result_path: str, out_path: str, k: int):
    try:
        df = pd.read_csv(result_path)
    except:
        print("Err: ", result_path)
        return np.zeros((k, 5))
    outputs = list[dict]()
    for gvals, record in zip(df["gvals"], df["record"]):
        for _, true, preds in ljson(record):
            row = {"gvals": gvals, "@k": k + 1, "resolved": 1, "budget": 0}
            if len(preds) == 0:
                row["resolved"] = 0
            else:
                row["budget"] = len(preds)
                for topk, pred in enumerate(preds, 1):
                    if true == pred:
                        row["@k"] = topk
                        break
            outputs.append(row)
    output_df = pd.DataFrame(outputs)
    resolved = output_df.query("resolved == 1")
    fail_to_pred = len(output_df) - len(resolved)
    summaries = []
    for topk in range(1, k + 1):
        pred_true = (resolved["@k"] <= topk).sum()
        pred_false = len(resolved) - pred_true
        pred_budget = (
            resolved[resolved["budget"] < topk]["budget"].sum()
            + (resolved["budget"] >= topk).sum() * topk
        )
        summaries.append(
            {
                "pred_true": pred_true,
                "pred_false": pred_false,
                "fail_to_pred": fail_to_pred,
                "total": len(output_df),
                "budget": pred_budget,
                "coverage": pred_true / len(output_df),
            }
        )

    output_df.to_csv(out_path, index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_path + ".summary.csv", index=False)
    return summary_df.loc[
        :, ["pred_true", "pred_false", "fail_to_pred", "total", "budget"]
    ].to_numpy()


def evaluate_report(result_dir: str, out_dir: str, k: int):
    summary = np.zeros((k, 5))
    for filename in tqdm(os.listdir(result_dir), desc="Evaluating"):
        summary += evaluate_report_chunk(
            os.path.join(result_dir, filename), os.path.join(out_dir, filename), k
        )
    df = pd.DataFrame(
        summary, columns=["pred_true", "pred_false", "fail_to_pred", "total", "budget"]
    )
    df["coverage"] = df["pred_true"] / df["total"]
    df.to_csv(out_dir + f"_{k}.csv")
