import copy
import io
import itertools
import os

import pandas as pd
import pytrec_eval as pe

from narraplay.documentranking.analysis.util import load_qrel_from_file
from narraplay.documentranking.config import QRELS_PATH, RUNS_DIR
from narraplay.documentranking.evaluate import extract_run
from narraplay.documentranking.rankers.ranker_weighted import run_weighted_ranker
from narraplay.documentranking.run_config import RANKER_BASE_MIN, BENCHMARKS, FIRST_STAGES, CONCEPT_STRATEGIES, \
    JUDGED_DOCS_ONLY_FLAG, WEIGHT_MATRIX

RESULT_MEASURES = {
    'recall_1000': 'Recall@1000',
    'ndcg_cut_10': 'nDCG@10',
    'ndcg_cut_20': 'nDCG@20',
    'ndcg_cut_100': 'nDCG@100',
    'P_10': 'P@10',
    'P_20': 'P@20',
    "P_100": 'P@100',
}


def main(ranker_base=RANKER_BASE_MIN, ablate_two=False, ablate_three=False):
    benchmark = BENCHMARKS[0]
    first_stage = FIRST_STAGES[0].name
    strategy = CONCEPT_STRATEGIES[0]

    ranker_bases = [ranker_base]
    weight_matrix = WEIGHT_MATRIX

    # prepare ablation data
    # first ignore for each round one of the rankers
    for i in range(len(ranker_base)):
        weights = [1 / (len(WEIGHT_MATRIX[0]) - 1)] * len(WEIGHT_MATRIX[0])
        weights[i] = 0.0
        weight_matrix.append(weights)

    if ablate_two:
        # second ignore for each round a pair of different rankers
        combinations = [*itertools.combinations(range(len(ranker_base)), 2)]
        for a, b in combinations:
            weights = [1 / 3] * 5
            weights[a] = 0.0
            weights[b] = 0.0
            weight_matrix.append(weights)

    if ablate_three:
        # third ignore for each round three different rankers
        combinations = [*itertools.combinations(range(len(ranker_base)), 3)]
        for a, b, c in combinations:
            weights = [1 / 2] * 5
            weights[a] = 0.0
            weights[b] = 0.0
            weights[c] = 0.0
            weight_matrix.append(weights)

    scores = run_weighted_ranker(benchmark=benchmark, weight_matrix=weight_matrix, ranker_bases=ranker_bases,
                                 strategy=strategy, first_stage=first_stage, return_results=True)

    qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
    qrel = load_qrel_from_file(str(qrel_path))

    relevant_topics = {str(t.query_id) for t in benchmark.topics}
    evaluator = pe.RelevanceEvaluator(qrel, [*RESULT_MEASURES.keys()], judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

    results = list()
    base_scores = [0.0] * len(RESULT_MEASURES)
    for config, weighted_run in scores.items():
        weights = [float(w) for w in config.split("-")[1:]]

        file = io.StringIO("\n".join(weighted_run))
        parsed_run = pe.parse_run(file)
        run_result = evaluator.evaluate(parsed_run)

        run_results = list()

        for measure in RESULT_MEASURES:
            run = extract_run(run_result, measure)

            # add missing scores
            for q in relevant_topics.difference(set(run.keys())):
                run.update({q: 0.0})

            # only evaluate relevant topics and skip the others
            run = {t: s for t, s in run.items() if t in relevant_topics}
            score = sum(run.values()) / len(run.keys())
            run_results.append(score)

        if all(w == (1.0 / len(WEIGHT_MATRIX[0])) for w in weights):
            # set weighted ranking run as base scoreline
            base_scores = copy.copy(run_results)
            run_results.insert(0, "Graph Reranking")
        elif any(w == (1.0 / (len(WEIGHT_MATRIX[0]) - 1)) for w in weights):
            # calculate differences to the base scoreline
            for i in range(len(base_scores)):
                run_results[i] = run_results[i] - base_scores[i]
            ranker_index = weights.index(0)
            name = ranker_base[ranker_index]
            run_results.insert(0, f"-{name.replace('DocumentRanker', '')} ({ranker_index})")
        else:
            # calculate differences to the base scoreline
            for i in range(len(base_scores)):
                run_results[i] = run_results[i] - base_scores[i]
            name = "|".join(str(i) for i in range(len(ranker_base)) if weights[i] == 0.0)
            run_results.insert(0, f"-({name.replace('DocumentRanker', '')})")

        results.append(run_results)
    df = pd.DataFrame(results, columns=["Ablation config", *RESULT_MEASURES.values()])
    df = df.round(2)

    print("==" * 60)
    print("Firststage    :", first_stage)
    print("Benchmark     :", benchmark.name)
    print("Ranker Base   :", [r.replace("DocumentRanker", "") for r in ranker_base])
    print("Weight matrix :", weight_matrix)
    print("==" * 60)
    print(df)
    print("==" * 60)

    print(df.to_latex())
    print(df.to_markdown())


if __name__ == "__main__":
    main()
