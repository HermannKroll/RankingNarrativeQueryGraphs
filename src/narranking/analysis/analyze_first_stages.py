import os

import pandas as pd
import pytrec_eval as pe

from narraplay.documentranking.analysis.util import load_qrel_from_file, load_run_from_file, extract_run, draw_bar_chart
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RUNS_DIR, RESULT_DIR_FIRST_STAGE, QRELS_PATH
from narraplay.documentranking.run_config import FIRST_STAGES, BENCHMARKS, CONCEPT_STRATEGIES, \
    JUDGED_DOCS_ONLY_FLAG, SKIPPED_TOPICS, FIRST_STAGE_TO_PRINT_NAME

MEASURES = {
    'recall_1000': 'Recall@1000',
    'set_recall': 'Recall',
    'num_ret': 'Retrieved',
    'P_5': 'P@5',
    'P_10': 'P@10',
    'P_20': 'P@20'
}

SUBDIR_PATH = "first_stages"


def generate_diagram(measures: list, strategy: str, benchmark: Benchmark, results: dict = None):
    rel_topics = {str(q.query_id) for q in benchmark.topics
                  if str(q.query_id) not in SKIPPED_TOPICS[benchmark.name]}

    for measure in measures:
        dfs = list()
        for config, run in results.items():
            run = extract_run(run, rel_topics, measure)
            df = pd.DataFrame.from_dict(run, orient="index", columns=[config])
            dfs.append(df)
        draw_bar_chart(dfs, SUBDIR_PATH, measure, benchmark.name, strategy)


def main():
    print('Generating diagramms...')
    for benchmark in BENCHMARKS:
        qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
        qrel = load_qrel_from_file(str(qrel_path))

        evaluator = pe.RelevanceEvaluator(qrel, [*MEASURES.keys()], judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

        for concept_strategy in CONCEPT_STRATEGIES:
            fs_results = dict()
            for first_stage in FIRST_STAGES:
                run_path = os.path.join(RESULT_DIR_FIRST_STAGE,
                                        f'{benchmark.name}_{first_stage.name}_{concept_strategy}.txt')
                run = load_run_from_file(run_path)

                fs_results[FIRST_STAGE_TO_PRINT_NAME[first_stage.name]] = evaluator.evaluate(run)
            generate_diagram(list(MEASURES.keys()), concept_strategy, benchmark, fs_results)
    print('Finished')

if __name__ == "__main__":
    main()
