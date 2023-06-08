import json
import os.path
import pickle

from tqdm import tqdm

from kgextractiontoolbox.backend.database import Session
from kgextractiontoolbox.backend.models import Predication
from narranking.benchmark import Benchmark
from narranking.config import MIN_SCORE_THRESHOLD, RESULT_DIR_FIRST_STAGE, QUERY_YIELD_PER_K
from narranking.config import CONCEPT_INDEX_PATH
from narranking.query import AnalyzedQuery
from narranking.translator import DocumentTranslator


class FirstStageRetriever:

    def __init__(self):
        self.graph_concept_index = None
        self.translator = DocumentTranslator()

        if not os.path.isfile(CONCEPT_INDEX_PATH):
            self.build_index()
        else:
            self.load_index()

    def build_index(self):
        self.graph_concept_index = {}

        session = Session.get()
        print('Creating graph concept index...')
        # iterate over all extracted statements
        total = session.query(Predication).count()
        print(f'Have to iterate over {total} statements...')
        query = session.query(Predication.document_collection, Predication.document_id,
                              Predication.subject_id, Predication.object_id)
        query = query.yield_per(QUERY_YIELD_PER_K)

        for r in tqdm(query, desc="Creating graph concept index", total=total):
            for c in [r.subject_id, r.object_id]:
                if r.document_collection not in self.graph_concept_index:
                    self.graph_concept_index[r.document_collection] = {}
                if c not in self.graph_concept_index[r.document_collection]:
                    self.graph_concept_index[r.document_collection][c] = set()
                self.graph_concept_index[r.document_collection][c].add(int(r.document_id))

        print(f'Graph concept index with {len(self.graph_concept_index)} key created')
        print(f'Writing index to file: {CONCEPT_INDEX_PATH}')
        with open(CONCEPT_INDEX_PATH, 'wb') as f:
            pickle.dump(self.graph_concept_index, f)
        print('Finished')

    def load_index(self):
        print(f'Loading graph concept index from {CONCEPT_INDEX_PATH}...')
        with open(CONCEPT_INDEX_PATH, 'rb') as f:
            self.graph_concept_index = pickle.load(f)
        print('Finished')

    def get_document_ids_for_concept(self, concept: str, collection: [str], id_filter: [int]):
        if collection in self.graph_concept_index:
            if concept in self.graph_concept_index[collection]:
                dids = self.graph_concept_index[collection][concept]
                # Apply document baseline id filter
                if collection == "PubMed" and id_filter:
                    dids = dids.intersection(id_filter)
                return self.translator.translate_document_ids_art2source(dids, collection)
        return set()

    def compute_concept_query_score(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):
        doc_id2score = {}
        part_score = 1.0 / len(query.component2concepts)
        statistics = {}
        document_ids_with_score = []
        score2count = {}
        if len(query.component2concepts) > 1:

            for idx, component in enumerate(query.component2concepts):
                document_ids_for_comp = set()
                for or_alt in query.component2concepts[component]:
                    document_ids_for_comp = document_ids_for_comp.union(self.get_document_ids_for_concept(or_alt,
                                                                                                          collection,
                                                                                                          benchmark.get_documents_for_baseline()))

                # Increment the score of all documents found by 1/len(comp)
                for did in document_ids_for_comp:
                    if did in doc_id2score:
                        # increment score
                        doc_id2score[did] += part_score
                    else:
                        doc_id2score[did] = part_score

                # keep track of some statistics
                statistics[component.strip()] = {
                    "concepts": query.component2concepts[component],
                    "document_ids": len(document_ids_for_comp)
                }

            # Put in list and sort
            document_ids_with_score = sorted([(k, v) for k, v in doc_id2score.items()], key=lambda x: x[1],
                                             reverse=True)

            # Count how often a score was assigned
            score2count = {}
            for d, score in document_ids_with_score:
                score_round = int(round(score * 100, 0))
                if score_round in score2count:
                    score2count[score_round] += 1
                else:
                    score2count[score_round] = 1
        else:
            statistics["query"] = "no_concept_found"

        statistics["all"] = {
            "document_ids": len(document_ids_with_score),
            "score2count": score2count
        }

        return document_ids_with_score, statistics


def main():
    BENCHMARKS = [
        Benchmark("trec-pm-2017-abstracts", "medline/2017/trec-pm-2017", ["PubMed", "trec-pm-201X-extra"]),
        Benchmark("trec-pm-2018-abstracts", "medline/2017/trec-pm-2018", ["PubMed", "trec-pm-201X-extra"]),
        Benchmark("trec-pm-2019-abstracts", "clinicaltrials/2019/trec-pm-2019", ["PubMed", "trec-pm-201X-extra"]),
        Benchmark("trec-pm-2017-cds", "clinicaltrials/2017/trec-pm-2017", ["trec-pm-2017"]),
        Benchmark("trec-pm-2018-cds", "clinicaltrials/2017/trec-pm-2018", ["trec-pm-2018-cds"]),
        Benchmark("trec-pm-2019-cds", "clinicaltrials/2019/trec-pm-2019", ["trec-pm-2019-cds"]),
        # Benchmark("trec-covid-rnd5", "cord19/trec-covid/round5", ["trec_covid_round5_abstract"]),
        Benchmark("trec-pm-2020-abstracts", "", ["PubMed"], load_from_file=True)
    ]

    stage = FirstStageRetriever()
    for bench in tqdm(BENCHMARKS):
        print('==' * 60)
        print('==' * 60)
        print(f'Running benchmark: {bench}')
        print('==' * 60)
        print('==' * 60)

        for concept_strategy in ["hybrid"]:  # ["exac", "expc", "hybrid"]:
            prefix = "FirstStageScore" + concept_strategy

            result_dir = RESULT_DIR_FIRST_STAGE
            stats_dir = os.path.join(result_dir, 'statistics')
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            statistics_data = {"used_min_score_threshold": MIN_SCORE_THRESHOLD}
            result_lines = []
            for idx, q in enumerate(bench.topics):
                print('--' * 60)
                print(f'Evaluating query {q}')
                print(f'Query components: {list(q.get_query_components())}')
                analyzed_query = AnalyzedQuery(q, concept_strategy=concept_strategy)

                ranked_docs = []
                for collection in bench.document_collections:
                    # print(f'Querying {collection} documents with: {fs_query}')
                    docs_for_c, statistics = stage.compute_concept_query_score(analyzed_query, collection, bench)
                    ranked_docs.extend(docs_for_c)
                    statistics_data[q.query_id] = statistics

                print(f'Received {len(ranked_docs)} documents')
                ranked_docs.sort(key=lambda x: (x[1], x[0]), reverse=True)
                count = 0
                for rank, (did, score) in enumerate(ranked_docs):
                    if score >= MIN_SCORE_THRESHOLD:
                        result_line = f'{q.query_id}\tQ0\t{did}\t{rank + 1}\t{score}\tFirstStageConceptScore'
                        result_lines.append(result_line)
                        count += 1
                print(f'{count} scored documents will be written (MIN_SCORE >= {MIN_SCORE_THRESHOLD})')

            path = os.path.join(result_dir, f'{bench.name}_FirstStageConceptScore.txt')
            print(f'Write ranked results to {path}')
            with open(path, 'wt') as f:
                f.write('\n'.join(result_lines))

            path = os.path.join(stats_dir, f'{bench.name}_{prefix}.json')
            print(f'Write statistics to {path}')
            with open(path, 'wt') as f:
                json.dump(statistics_data, f, indent=2)
            print('--' * 60)


if __name__ == "__main__":
    main()
