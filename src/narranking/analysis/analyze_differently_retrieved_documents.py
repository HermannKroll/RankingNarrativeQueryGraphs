import os
from typing import List

import pytrec_eval as pe

from narraplay.documentranking.analysis.util import load_qrel_from_file, load_run_from_file
from narraplay.documentranking.config import RUNS_DIR, RESULT_DIR_FIRST_STAGE, QRELS_PATH, RESULT_DIR
from narraplay.documentranking.run_config import FIRST_STAGES, BENCHMARKS, CONCEPT_STRATEGIES, JUDGED_DOCS_ONLY_FLAG

MEASURES = {
    'P_10': 'P@10',
    'set_recall': 'Recall@All'
}

EVALUATED_STRATEGY = "WeightedDocumentRanker_min-0.25-0.25-0.25-0.25"
TOP_K = 30


def get_top_k_scored_documents(documents2score: dict, k: int) -> List[str]:
    doc_list = [(d, score) for d, score in documents2score.items()]
    doc_list.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in doc_list[:k]]


def main():
    documents_to_review = list()

    for benchmark in BENCHMARKS:
        topic2string = {t.query_id: list(t.get_query_components()) for t in benchmark.topics}
        qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
        qrel = load_qrel_from_file(str(qrel_path))

        evaluator = pe.RelevanceEvaluator(qrel, [*MEASURES.keys()], judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

        for concept_strategy in CONCEPT_STRATEGIES:
            for first_stage in FIRST_STAGES:
                run_path_root = os.path.join(RESULT_DIR, f"{first_stage.name}_{concept_strategy}")
                run_path = os.path.join(run_path_root, f'{benchmark.name}_{EVALUATED_STRATEGY}.txt')
                run = load_run_from_file(run_path)
                scores = evaluator.evaluate(run)
                print('--' * 60)
                print(f'Evaluated run: {run_path}')
                for topic in run:
                    missing_documents = list()
                    wrongly_retrieved = list()
                    correctly_retrieved = list()
                    for docid in get_top_k_scored_documents(run[topic], TOP_K):
                        if docid not in qrel[topic]:
                            missing_documents.append(docid)
                        elif docid in qrel[topic] and qrel[topic][docid] == 0:
                            wrongly_retrieved.append(docid)
                        elif docid in qrel[topic] and qrel[topic][docid] in [1, 2]:
                            correctly_retrieved.append(docid)

                    documents_to_review.extend([(d, topic, topic2string[topic]) for d in missing_documents])
                    print(f'Topic: {topic} ({topic2string[topic]})')
                    print(f'Retrieved number of documents: {len(run[topic])}')
                    print(f'Scores : {scores[topic]}')
                    print(f'Retrieved unrated documents: {missing_documents}')
                    print(f'Wrongly retrieved documents: {wrongly_retrieved}')
                    print(f'Correctly retrieved documents: {correctly_retrieved}')
                    print()
                print('--' * 60)

    print(f'{len(documents_to_review)} documents to review:')
    for entry in documents_to_review:
        print(entry)


if __name__ == "__main__":
    main()
