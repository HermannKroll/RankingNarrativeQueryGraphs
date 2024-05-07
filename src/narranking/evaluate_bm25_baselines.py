import os

import pytrec_eval as pe

from narraplay.documentranking.config import RUNS_DIR, QRELS_PATH, \
    RESULT_DIR_BASELINES
from narraplay.documentranking.evaluate import load_qrel_from_file, load_run_from_file, MEASURES, save_results, \
    generate_table, RESULT_MEASURES
from narraplay.documentranking.first_stages.first_stage_graph import FirstStageGraphRetriever
from narraplay.documentranking.run_config import BENCHMARKS, JUDGED_DOCS_ONLY_FLAG


def evaluate_runs():
    print("Starting evaluation")
    for benchmark in BENCHMARKS:
        qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
        qrel = load_qrel_from_file(str(qrel_path))

        evaluator = pe.RelevanceEvaluator(qrel, MEASURES, judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)
        ranker_results = dict()

        bm25_run_path = os.path.join(RESULT_DIR_BASELINES, f'{benchmark.name}_BM25.txt')
        bm25_run = load_run_from_file(bm25_run_path)
        ranker_results["BM25 Native"] = evaluator.evaluate(bm25_run)

        # hack required to filter for translatable topics
        save_results(ranker_results, "likesimilarity", FirstStageGraphRetriever().name, benchmark.name)


if __name__ == "__main__":
    print('\n\n\n')
    evaluate_runs()
    for benchmark in BENCHMARKS:
        print("--" * 60)
        print(f'Benchmark:      : {benchmark.name}')
        print(f'Concept strategy: BM25 Native')
        print("--" * 60)
        # hack required to filter for translatable topics
        generate_table(RESULT_MEASURES, "likesimilarity", [FirstStageGraphRetriever().name], benchmark)
