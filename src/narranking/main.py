import json
import multiprocessing
import os.path
from datetime import datetime

from tqdm import tqdm

from kgextractiontoolbox.util.multiprocessing.ConsumerWorker import ConsumerWorker
from kgextractiontoolbox.util.multiprocessing.ProducerWorker import ProducerWorker
from kgextractiontoolbox.util.multiprocessing.Worker import Worker
from narranking.benchmark import Benchmark
from narranking.config import RESULT_DIR, RESULT_DIR_TEMP, RESULT_DIR_FIRST_STAGE, IGNORE_DEMOGRAPHIC
from narranking.corpus import DocumentCorpus
from narranking.query import AnalyzedQuery
from narranking.rankers.graph import PathFrequencyRankerWeightedOld, PathsFrequencyRankerWeighted, \
    PathsFrequencyInverseLengthRankerWeighted, PathsFrequencyLengthRankerWeighted, ShortestPathLengthRankerWeighted, \
    ShortestPathInverseLengthRankerWeighted, ShortestPathFrequencyRankerWeighted, ShortestPathConfidenceRankerWeighted
from narranking.rankers.statement import StatementPartialOverlapDocumentRankerWeighted, \
    StatementPartialOverlapFrequencyDocumentRankerWeighted, StatementPartialOverlapConfidenceDocumentRankerWeighted, \
    StatementOverlapDocumentRankerWeighted, StatementOverlapFrequencyDocumentRankerWeighted, \
    StatementOverlapConfidenceDocumentRankerWeighted
from narranking.retriever import DocumentRetriever

RETRIEVE_DOCUMENT_DATA = True
COMPUTE_RANKING_RESULTS = True
MINIMUM_CONFIDENCE_THRESHOLDS = [0, 0.3, 0.6]

print('=' * 60)
print(f'Demographic Ignored: {IGNORE_DEMOGRAPHIC}')
print('=' * 60)

BENCHMARKS = [
    Benchmark("trec-pm-2020-abstracts", "", ["PubMed"], load_from_file=True),
    Benchmark("trec-pm-2017-abstracts", "medline/2017/trec-pm-2017", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2018-abstracts", "medline/2017/trec-pm-2018", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2019-abstracts", "clinicaltrials/2019/trec-pm-2019", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2017-cds", "clinicaltrials/2017/trec-pm-2017", ["trec-pm-2017"]),
    Benchmark("trec-pm-2018-cds", "clinicaltrials/2017/trec-pm-2018", ["trec-pm-2018-cds"]),
    Benchmark("trec-pm-2019-cds", "clinicaltrials/2019/trec-pm-2019", ["trec-pm-2019-cds"])
    # Benchmark("trec-covid-rnd5", "cord19/trec-covid/round5", ["trec_covid_round5_abstract"]),
]

RANKING_STRATEGIES = [  # EqualDocumentRanker(), ConceptDocumentRanker(), ConceptFrequencyDocumentRanker(),
    # StatementPartialOverlapDocumentRanker(), StatementOverlapDocumentRanker(),
    # TagFrequencyRanker(), StatementFrequencyRanker(), ConfidenceStatementFrequencyRanker(),
    # PathFrequencyRanker(), ConfidencePathFrequencyRanker(), AdjacentEdgesRanker(),
    # ConfidenceAdjacentEdgesRanker(), BM25Tag(),
    # weighted ranker
    # ConceptDocumentRankerWeighted(),
    # ConceptFrequencyDocumentRankerWeighted(),

    StatementPartialOverlapDocumentRankerWeighted(),
    StatementPartialOverlapFrequencyDocumentRankerWeighted(),
    StatementPartialOverlapConfidenceDocumentRankerWeighted(),

    StatementOverlapDocumentRankerWeighted(),
    StatementOverlapFrequencyDocumentRankerWeighted(),
    StatementOverlapConfidenceDocumentRankerWeighted(),

    PathFrequencyRankerWeightedOld(),
    PathsFrequencyRankerWeighted(),
    PathsFrequencyLengthRankerWeighted(),
    PathsFrequencyInverseLengthRankerWeighted(),

    ShortestPathLengthRankerWeighted(),
    ShortestPathInverseLengthRankerWeighted(),
    ShortestPathFrequencyRankerWeighted(),
    ShortestPathConfidenceRankerWeighted()

    # TagFrequencyRankerWeighted(),
    # StatementFrequencyRankerWeighted(),
    # ConfidenceStatementFrequencyRankerWeighted(),
    # PathFrequencyRankerWeighted(),
    # ConfidencePathFrequencyRankerWeighted(),
    # AdjacentEdgesRankerWeighted(),
    # ConfidenceAdjacentEdgesRankerWeighted(),
    # BM25TagWeighted()
]


def load_document_ids_from_runfile(path_to_runfile):
    topic2docs = {}
    with open(path_to_runfile, 'rt') as f:
        for line in f:
            components = line.split('\t')
            topic_id = int(components[0])
            doc_id = components[2]

            if topic_id not in topic2docs:
                topic2docs[topic_id] = [doc_id]
            else:
                topic2docs[topic_id].append(doc_id)

    return topic2docs


FIRST_STAGES = ["FirstStageConceptScore"]
#["FirstStageBM25Own", "FirstStageBM25", "FirstStageConceptScore"]
# "BenchmarkRuns"]
#  "FirstStageBM25"]# ,
#  "FirstStageConceptScore"]

for bench in tqdm(BENCHMARKS):
    retriever = DocumentRetriever()
    print('==' * 60)
    print('==' * 60)
    print(f'Running benchmark: {bench}')
    print('==' * 60)
    print('==' * 60)
    for first_stage in FIRST_STAGES:
        for concept_strategy in ["hybrid"]:  # ["exac", "expc", "hybrid"]:
            prefix = concept_strategy + '_' + first_stage
            stats_dir = os.path.join(RESULT_DIR, 'statistics')
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            result_dir = os.path.join(RESULT_DIR, prefix)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            result_dir_temp = os.path.join(RESULT_DIR_TEMP, prefix)
            if not os.path.exists(result_dir_temp):
                os.makedirs(result_dir_temp)

            statistics_data = []

            topic2ids = {}

            if first_stage in ["FirstStageBM25", "FirstStageConceptScore", "FirstStageBM25Own"]:
                path = os.path.join(RESULT_DIR_FIRST_STAGE, f'{bench.name}_{first_stage}.txt')
                topic2ids = load_document_ids_from_runfile(path)
            elif first_stage in ["BenchmarkRuns"]:
                pass
            else:
                raise ValueError(f'First stage mode {first_stage} not supported at the moment')

            for idx, q in enumerate(bench.topics):
                print('--' * 60)
                print(f'Evaluating query {q}')
                print(f'Query components: {list(q.get_query_components())}')
                analyzed_query = AnalyzedQuery(q, concept_strategy=concept_strategy)
                analyzed_query_statistics = analyzed_query.get_statistics()

                if RETRIEVE_DOCUMENT_DATA:
                    if first_stage == "BenchmarkRuns":
                        doc_ids_to_rank = bench.topic_id2docs[q.query_id]
                    else:
                        if int(q.query_id) in topic2ids:
                            doc_ids_to_rank = topic2ids[int(q.query_id)]
                        else:
                            doc_ids_to_rank = set()
                    print(f'Querying for {len(doc_ids_to_rank)} documents...')
                    narrative_docs = list(retriever.retrieve_narrative_documents_for_collections(doc_ids_to_rank,
                                                                                                 bench.document_collections))

                    narrative_docs = sorted(narrative_docs, key=lambda x: x.document_id_source)
                    print(f'{len(narrative_docs)} documents retrieved')

                    if len(narrative_docs) > 0:
                        for confidence in MINIMUM_CONFIDENCE_THRESHOLDS:
                            for d in narrative_docs:
                                d.prepare_with_min_confidence(confidence)
                            analyzed_query_statistics["c@{}".format(int(confidence * 100))] = \
                                analyzed_query.get_document_statistics(narrative_docs)

                            if COMPUTE_RANKING_RESULTS:
                                corpus = DocumentCorpus(narrative_docs)
                                print('--' * 60)
                                print('Starting ranking phase... [confidence>={}]'.format(confidence))


                                def generate_tasks():
                                    for ranker in RANKING_STRATEGIES:
                                        yield ranker


                                def do_task(ranker):
                                    start = datetime.now()
                                    w_results = []
                                    ranked_docs = ranker.rank_documents(analyzed_query, narrative_docs, corpus)
                                    if len(ranked_docs) > 0:
                                        max_score = ranked_docs[0][1]
                                        for rank, (doc_id, score) in enumerate(ranked_docs):
                                            if max_score > 0:
                                                norm_score = score / max_score
                                            else:
                                                norm_score = 0.0
                                            result_line = f'{q.query_id}\tQ0\t{doc_id}\t{rank + 1}\t{norm_score}\t{ranker.name}'
                                            w_results.append(result_line)
                                    time_taken = datetime.now() - start
                                    print(f'{time_taken}s to compute {ranker.name}')
                                    return ranker.name, w_results


                                def consume_task(worker_res):
                                    ranker_name, result_lines = worker_res
                                    path = os.path.join(result_dir_temp,
                                                        f'{bench.name}_{q.query_id}_{ranker_name}_c{int(confidence * 100)}.txt')
                                    with open(path, 'wt') as f:
                                        f.write('\n'.join(result_lines))


                                no_workers = len(RANKING_STRATEGIES)
                                task_queue = multiprocessing.Queue()
                                result_queue = multiprocessing.Queue()
                                producer = ProducerWorker(task_queue, generate_tasks, no_workers, max_tasks=100000)
                                workers = [Worker(task_queue, result_queue, do_task) for n in range(no_workers)]
                                consumer = ConsumerWorker(result_queue, consume_task, no_workers)

                                producer.start()
                                for w in workers:
                                    w.start()
                                consumer.start()
                                consumer.join()
                statistics_data.append(analyzed_query_statistics)

            if COMPUTE_RANKING_RESULTS:
                print('--' * 60)
                print('Finalizing result file...')
                for r in RANKING_STRATEGIES:
                    for c in MINIMUM_CONFIDENCE_THRESHOLDS:
                        results = []
                        for q in bench.topics:
                            path_to_temp_file = os.path.join(result_dir_temp,
                                                             f'{bench.name}_{q.query_id}_{r.name}_c{int(c * 100)}.txt')
                            if os.path.isfile(path_to_temp_file):
                                with open(path_to_temp_file, 'rt') as f:
                                    for line in f:
                                        results.append(line.strip())
                                os.remove(path_to_temp_file)

                        path = os.path.join(result_dir, f'{bench.name}_{r.name}_c{int(c * 100)}.txt')
                        with open(path, 'wt') as f:
                            f.write('\n'.join(results))

            path = os.path.join(stats_dir, f'{bench.name}_{prefix}.json')
            with open(path, 'wt') as f:
                statistics = dict()
                for obj in statistics_data:
                    cnt = 0
                    for _, concepts in obj["component2concepts"].items():
                        length = len(concepts)
                        if length > 0:
                            cnt += 1
                    if cnt not in statistics:
                        statistics[cnt] = 0
                    statistics[cnt] += 1

                json.dump(dict(data=statistics_data, statistics=statistics), f, indent=2)
            print('--' * 60)

# Write results
# results/benchmark_name/metric.txt
# QueryTopic Bla Doc_ID Rang Score Metric.Name s

# 1	Q0	v861kk0i	1	45602.81753540039	BioInfo-run1
