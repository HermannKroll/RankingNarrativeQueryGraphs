import io
import itertools
import json
import os
from datetime import datetime

import pytrec_eval as pe
import tqdm

from narraplay.documentranking.config import RUNS_DIR, QRELS_PATH, RESULT_DIR_HYPERPARAMS
from narraplay.documentranking.evaluate import load_qrel_from_file
from narraplay.documentranking.rankers.ranker_weighted import run_weighted_ranker
from narraplay.documentranking.run_config import BENCHMARKS, RANKER_BASES, JUDGED_DOCS_ONLY_FLAG, CONCEPT_STRATEGIES, \
    FIRST_STAGES


def weight_combinations(vector_length=5, step_size=0.1, max_weight=0.9, target_sum=1.0):
    max_step_size = int(max_weight / step_size)
    vector_values = [round(i * step_size, 2) for i in range(0, max_step_size)]  # Values from 0.1 to 0.9

    print("==" * 60)
    print("Combination config")
    print("vector_length:", vector_length)
    print("step_size    :", step_size)
    print("max_step_size:", max_step_size)
    print("max_weight   :", max_weight)
    print("target_sum   :", target_sum)
    print("==" * 60)

    combinations = itertools.product(vector_values, repeat=vector_length)
    valid_combinations = [combo for combo in combinations if sum(combo) == target_sum]

    return valid_combinations


def highest_score_from_file():
    benchmark = BENCHMARKS[0]
    strategy = CONCEPT_STRATEGIES[0]

    for first_stage in FIRST_STAGES:
        print(f"{first_stage.name}_{strategy}_{benchmark.name}.json")
        file_name = os.path.join(RESULT_DIR_HYPERPARAMS, f"{first_stage.name}_{strategy}_{benchmark.name}.json")
        if not os.path.isfile(file_name):
            raise FileNotFoundError("The combination of parameters is not evaluated yet.", file_name)
        with open(file_name, "rt") as file:
            results = json.load(file)

        score_max = 0.0
        score_config = ""
        for config, score in results.items():
            if score > score_max:
                score_max = score
                score_config = config
        config = score_config.split("-")

        print("Using file", file_name)
        print("Best weight combination reached a score of {} for {}".format(round(score_max, 2), first_stage.name))
        print("Weighted ranker type: {}".format(config[0]))
        print("Weights combinations: [{}, {}, {}, {}, {}]".format(*[round(float(s), 2) for s in config[1:]]))
        print("==" * 60)


def main(judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG):
    start = datetime.now()
    weight_matrix = weight_combinations(vector_length=5, step_size=0.05, max_weight=1.0, target_sum=1.0)

    print("Trying {} * {} = {} combinations."
          .format(len(weight_matrix), len(RANKER_BASES), len(weight_matrix) * len(RANKER_BASES)))

    # initial configuration for hyperparameter search
    # GraphRetriever + concept strategy: likesimilarity
    benchmark = BENCHMARKS[0]
    strategy = CONCEPT_STRATEGIES[0]
    first_stage = FIRST_STAGES[0].name
    target_measure = ["P_20"]

    # run weighted ranker on all combos
    scores = run_weighted_ranker(benchmark=benchmark, weight_matrix=weight_matrix, ranker_bases=RANKER_BASES,
                                 strategy=strategy, first_stage=first_stage, return_results=True)

    print("==" * 60)

    qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
    qrel = load_qrel_from_file(str(qrel_path))

    relevant_topics = {str(t.query_id) for t in benchmark.topics}

    evaluator = pe.RelevanceEvaluator(qrel, target_measure, judged_docs_only_flag=judged_docs_only_flag)
    ranker_results = dict()

    print("Relevant topics: ", relevant_topics)
    print("==" * 60)
    print("Begin evaluation process")

    for config, results in tqdm.tqdm(scores.items(), total=len(scores.keys())):
        # prepare run file from weighted ranker results
        file = io.StringIO("\n".join(results))
        run = pe.parse_run(file)
        eval_result = evaluator.evaluate(run)

        # extract scores from results and add missing topics as 0.0 scores
        scores = {q_id: m_score[target_measure[0]] for q_id, m_score in eval_result.items() if q_id in relevant_topics}
        scores.update([(q_id, 0.0) for q_id in relevant_topics.difference([*scores.keys()])])

        # calculate mean score
        score = sum(scores.values()) / len(relevant_topics)
        ranker_results[config] = score

    print("==" * 60)
    file_name = os.path.join(RESULT_DIR_HYPERPARAMS, f"{first_stage}_{strategy}_{benchmark.name}.json")
    print("finished... storing into file", file_name)
    with open(file_name, "wt") as file:
        json.dump(ranker_results, file, indent=2)
    time_taken = datetime.now() - start
    print(f'{time_taken}s to compute ranker results')
    print("==" * 60)


if __name__ == "__main__":
    # main(judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)
    highest_score_from_file()
