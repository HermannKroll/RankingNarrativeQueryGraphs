import os

import pytrec_eval as pe

from narraplay.documentranking.analysis.analyze_first_stages import MEASURES, generate_diagram
from narraplay.documentranking.analysis.util import load_qrel_from_file, load_run_from_file
from narraplay.documentranking.config import RUNS_DIR, RESULT_DIR_FIRST_STAGE, QRELS_PATH
from narraplay.documentranking.run_config import FIRST_STAGES, BENCHMARKS, CONCEPT_STRATEGIES, \
    JUDGED_DOCS_ONLY_FLAG, CONCEPT_STRATEGY_2_REAL_NAME, FIRST_STAGE_TO_PRINT_NAME

SKIPPED = [
    "Full Match + LIKE + Ontology"
]


def main():
    print('Generating diagramms...')
    for benchmark in BENCHMARKS:
        qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
        qrel = load_qrel_from_file(str(qrel_path))

        evaluator = pe.RelevanceEvaluator(qrel, [*MEASURES.keys()], judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

        fs_results = dict()
        for concept_strategy in CONCEPT_STRATEGIES:

            for first_stage in FIRST_STAGES:
                run_path = os.path.join(RESULT_DIR_FIRST_STAGE,
                                        f'{benchmark.name}_{first_stage.name}_{concept_strategy}.txt')
                run = load_run_from_file(run_path)

                n = f'{FIRST_STAGE_TO_PRINT_NAME[first_stage.name]} + {CONCEPT_STRATEGY_2_REAL_NAME[concept_strategy]}'
                if n not in SKIPPED:
                    fs_results[n] = evaluator.evaluate(run)
        generate_diagram(list(MEASURES.keys()), "combined", benchmark, fs_results)
    print('Finished')


if __name__ == "__main__":
    main()
