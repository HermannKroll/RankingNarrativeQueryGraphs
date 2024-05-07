import os
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple

from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RESULT_DIR, RESULT_DIR_FIRST_STAGE
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.run_config import WEIGHT_MATRIX, RANKER_BASES, CONCEPT_STRATEGIES, FIRST_STAGE_NAMES, \
    BENCHMARKS


class WeightedDocumentRanker:
    def __init__(self, benchmark, first_stage, strategy, name="WeightedDocumentRanker"):
        self.name = name
        self.benchmark = benchmark
        self.first_stage = first_stage
        self.strategy = strategy
        self.rankers = list()
        self.weights = list()
        self.results = dict()

    @staticmethod
    def _read_ranker_results(filename):
        with open(filename, 'r') as f:
            data = defaultdict(dict)
            for line in f.readlines():
                # 39 Q0 26939704 4 0.6666666666666665 ConfidenceDocumentRanker
                q, _, doc_id, _, score, _ = line.strip().split('\t')
                data[q][doc_id] = float(score)
            return data

    def _load_dependencies(self):
        for ranker in self.rankers:
            path = os.path.join(RESULT_DIR, f"{self.first_stage}_{self.strategy}", f"{self.benchmark}_{ranker}.txt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Result file for {ranker} not found")
            self.results[ranker] = self._read_ranker_results(path)

    def set_first_stage(self, first_stage):
        if first_stage == self.first_stage:
            return
        self.first_stage = first_stage
        self.results = dict()
        self._load_dependencies()

    def set_rankers(self, rankers):
        if rankers == self.rankers:
            return
        self.rankers = rankers
        self.results = dict()
        self._load_dependencies()

    def set_weights(self, weights: List[float]):
        if len(weights) != len(self.rankers):
            raise ValueError(f"Expected an equal number of weights and rankers, but got {len(weights)} and {len(self.rankers)}")
        if sum(weights) != 1.0:
            raise ValueError(f"Expected sum of weights to be 1.0 but got {sum(weights)}")
        self.weights = weights

    def configuration(self):
        if "Only" in self.rankers[1]:
            return "-".join(["only", *(str(w) for w in self.weights)])
        if "Plus" in self.rankers[1]:
            return "-".join(["plus", *(str(w) for w in self.weights)])
        if "Avg" in self.rankers[1]:
            return "-".join(["avg", *(str(w) for w in self.weights)])
        if "Min" in self.rankers[1]:
            return "-".join(["min", *(str(w) for w in self.weights)])
        if "Max" in self.rankers[1]:
            return "-".join(["max", *(str(w) for w in self.weights)])
        raise NotImplementedError("The given ranker is unknown")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[str]) -> List[Tuple[str, float]]:
        if not self.weights:
            raise ValueError("Call set_weights() first.")
        scores = list()
        for doc_id in narrative_documents:
            q_id = query.topic.query_id

            sub_scores = []
            for w, r in zip(self.weights, self.rankers):
                if doc_id not in self.results[r][q_id]:
                    print(f'Warning: document id {doc_id} is missing in ranker {r}')
                    continue

                sub_scores.append(self.results[r][q_id][doc_id] * w)

            score = sum(sub_scores)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scores


def load_document_ids_from_runfile(path_to_runfile):
    topic2docs = {}
    with open(path_to_runfile, 'rt') as f:
        for line in f:
            components = line.split('\t')
            topic_id = int(components[0])
            doc_id = components[2]
            score = float(components[4])

            if topic_id not in topic2docs:
                topic2docs[topic_id] = [(doc_id, score)]
            else:
                topic2docs[topic_id].append((doc_id, score))
    return topic2docs


def run_weighted_ranker(benchmark: Benchmark, weight_matrix: List[List[float]], ranker_bases: List[List[str]], strategy,
                        first_stage, return_results=False):
    prefix = first_stage + '_' + strategy
    result_dir = os.path.join(RESULT_DIR, prefix)

    ranker = WeightedDocumentRanker(benchmark.name, first_stage, strategy)

    path = os.path.join(RESULT_DIR_FIRST_STAGE, f'{benchmark.name}_{first_stage}_{strategy}.txt')
    topic2ids = load_document_ids_from_runfile(path)

    start = datetime.now()
    result_lines = defaultdict(list)

    for ranker_base in ranker_bases:
        ranker.set_rankers(ranker_base)
        print(f'Run ranker base: {ranker_base}')

        for idx, q in enumerate(benchmark.topics):
            if int(q.query_id) in topic2ids:
                fs_docs_with_scores = topic2ids[int(q.query_id)]
            else:
                fs_docs_with_scores = list()
            fs_doc_ids = [d[0] for d in fs_docs_with_scores]

            print('--' * 60)
            print("Current topic", str(q))
            analyzed_query = AnalyzedQuery(q, concept_strategy=strategy, translate_query=False)

            for weights in weight_matrix:
                ranker.set_weights(weights)
                ranked_docs = ranker.rank_documents(analyzed_query, fs_doc_ids)
                run_config = ranker.configuration()

                if len(ranked_docs) > 0:
                    for rank, (doc_id, score) in enumerate(ranked_docs):
                        if score > 1.0 or score < 0.0:
                            raise ValueError(
                                f'Document {doc_id} received a score not in [0, 1] (score = {score})')

                        result_line = f'{q.query_id}\tQ0\t{doc_id}\t{rank + 1}\t{score}\t{ranker.name}'
                        result_lines[run_config].append(result_line)

    print('--' * 60)
    time_taken = datetime.now() - start
    print(f'{time_taken}s to compute {ranker.name}')

    if return_results:
        return result_lines
    print('Finalizing result file...')

    for run_config in result_lines:
        path = os.path.join(result_dir, f'{benchmark.name}_{ranker.name}_{run_config}.txt')
        print(f"{run_config} written to {path}")
        # print("{}".format('\n'.join(result_lines[run_config][:10])))
        with open(path, 'wt') as f:
            f.write('\n'.join(result_lines[run_config]))


if __name__ == "__main__":
    for bench in BENCHMARKS:
        for first_stage in FIRST_STAGE_NAMES:
            for strategy in CONCEPT_STRATEGIES:
                print("--" * 60)
                print(f'Benchmark       : {bench.name}')
                print(f'First stage     : {first_stage}')
                print(f'Concept strategy: {strategy}')
                print("--" * 60)
                run_weighted_ranker(benchmark=bench, weight_matrix=WEIGHT_MATRIX, ranker_bases=RANKER_BASES,
                                    strategy=strategy, first_stage=first_stage)
