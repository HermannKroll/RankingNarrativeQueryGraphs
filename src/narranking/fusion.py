import os
from collections import defaultdict

import pandas as pd
from ranx import Qrels, Run, compare, fuse

from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RESULT_DIR, RUNS_DIR, EVAL_DIR, QRELS_PATH
from narraplay.documentranking.run_config import BENCHMARKS, FIRST_STAGES, CONCEPT_STRATEGIES, \
    RANKER_BASES


def load_run(path):
    try:
        return Run.from_file(path, kind="trec")
    except:
        # catch invalid ranx.Run-formatted runs (e.g. uncompressed/trec-pm-2019-abstracts/SIBTMlit3)
        runs_dict = defaultdict(dict)
        with open(path, "rt") as file:
            for line in file.readlines():
                q_id, _, doc_id, _, rel, _ = line.split()[:6]
                runs_dict[q_id][doc_id] = float(rel)
        return Run.from_dict(runs_dict)


def rank_fusion(track: Benchmark, first_stage, strategy: str, baseline: str, fusion_rankers: [str], metrics: [str]):
    qrels = Qrels.from_file(os.path.join(RUNS_DIR, QRELS_PATH[track.name]), kind="trec")

    baseline_path = os.path.join(RESULT_DIR, f"{first_stage}_{strategy}", f"{track.name}_{baseline}.txt")
    if not os.path.isfile(baseline_path):
        print(f"Skipped config {baseline_path}")
        return
    baseline_run = load_run(baseline_path)
    baseline_run.make_comparable(qrels)

    run_paths = [os.path.join(RESULT_DIR, f"{first_stage}_{strategy}", f"{track.name}_{rp}.txt") for rp in fusion_rankers]

    # ranker runs
    runs = []
    for run_path in run_paths:
        run = load_run(run_path)
        run.make_comparable(qrels)
        runs.append(run)

    # run fusion on baseline using every ranking strategy
    final_run = fuse(runs=[baseline_run, *runs], method='rrf')
    report = compare(
        qrels=qrels,
        runs=[baseline_run, final_run],
        metrics=metrics,
        rounding_digits=4,
        max_p=0.05,
    )

    print("--" * 60)
    print(f"Benchmark    : {benchmark.name}")
    print(f"FirstStage   : {first_stage}")
    print(f"Strategy     : {strategy}")
    print(f"Baseline     : {baseline}")
    print(f"FusionRankers: {', '.join(fusion_rankers)}")
    print(report.to_table())
    print("--" * 60)

    fusion_results = {}
    for m in metrics:
        fusion_results[m] = final_run.mean_scores.get(m) - baseline_run.mean_scores.get(m)

    # TfIdf ranker kind
    prefix = fusion_rankers[1][5:8].lower()
    result_path = os.path.join(EVAL_DIR, f"{first_stage}_{strategy}_", f"{track.name}", f"{prefix}_fusion_results.json")

    try:
        # may fail due to permission denied exceptions
        # csv data export
        df = pd.DataFrame.from_dict(fusion_results, orient="index").T
        df.to_csv(result_path)
    except:
        pass


METRICS = [
    #"recall",
    "ndcg@1000",
    "map@1000",
    #"precision@10",
    "precision@20",
    #"precision@100",
    #"bpref"
]


if __name__ == "__main__":
    for benchmark in BENCHMARKS:
        for first_stage in FIRST_STAGES:
            for strategy in CONCEPT_STRATEGIES:
                for fusion_rankers in RANKER_BASES:
                    rank_fusion(benchmark, first_stage, strategy, "BM25Text", fusion_rankers, METRICS)
