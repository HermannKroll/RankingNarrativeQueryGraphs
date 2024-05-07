import os

import pytrec_eval as pe

from narraplay.documentranking.analysis.util import load_qrel_from_file, load_run_from_file
from narraplay.documentranking.config import RUNS_DIR, RESULT_DIR_FIRST_STAGE, QRELS_PATH
from narraplay.documentranking.run_config import FIRST_STAGES, BENCHMARKS, CONCEPT_STRATEGIES, JUDGED_DOCS_ONLY_FLAG

MEASURES = {
    'P_10': 'P@10',
    'set_recall': 'Recall@All'
}


def main():
    for benchmark in BENCHMARKS:
        topic2string = {t.query_id: list(t.get_query_components()) for t in benchmark.topics}
        qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
        qrel = load_qrel_from_file(str(qrel_path))

        evaluator = pe.RelevanceEvaluator(qrel, [*MEASURES.keys()], judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

        for concept_strategy in CONCEPT_STRATEGIES:
            for first_stage in FIRST_STAGES:
                run_path = os.path.join(RESULT_DIR_FIRST_STAGE,
                                        f'{benchmark.name}_{first_stage.name}_{concept_strategy}.txt')
                run = load_run_from_file(run_path)
                scores = evaluator.evaluate(run)
                print('--' * 60)
                print(f'Evaluated run: {run_path}')
                for topic in run:
                    missing_documents = list()
                    wrongly_retrieved = list()
                    correctly_retrieved = list()
                    for docid in run[topic]:
                        if docid not in qrel[topic]:
                            missing_documents.append(docid)
                        elif docid in qrel[topic] and qrel[topic][docid] == 0:
                            wrongly_retrieved.append(docid)
                        elif docid in qrel[topic] and qrel[topic][docid] in [1, 2]:
                            correctly_retrieved.append(docid)


                    print(f'Topic: {topic} ({topic2string[topic]})')
                    print(f'Retrieved number of documents: {len(run[topic])}')
                    print(f'Scores : {scores[topic]}')
                    print(f'Retrieved unrated documents: {missing_documents}')
                    print(f'Wrongly retrieved documents: {wrongly_retrieved}')
                    print(f'Correctly retrieved documents: {correctly_retrieved}')
                    print()
                print('--' * 60)


if __name__ == "__main__":
    main()
