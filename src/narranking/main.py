import json
import logging
import os.path
from datetime import datetime

from tqdm import tqdm

from narraplay.documentranking.config import RESULT_DIR, RESULT_DIR_FIRST_STAGE
from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.graph_fragment import GraphFragment
from narraplay.documentranking.rankers.ranker_weighted import run_weighted_ranker
from narraplay.documentranking.retriever import DocumentRetriever
from narraplay.documentranking.run_config import BENCHMARKS, FIRST_STAGE_NAMES, CONCEPT_STRATEGIES, WEIGHT_MATRIX, \
    RANKING_STRATEGIES, RANKER_BASES, IGNORE_DEMOGRAPHIC

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

print('=' * 60)
print(f'Demographic Ignored: {IGNORE_DEMOGRAPHIC}')
print('=' * 60)


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


def compute_first_stage_scores(fs_docs_with_scores):
    fs_doc_id2upper_bound = {str(d[0]): d[1] for d in fs_docs_with_scores}
    fs_doc_distinct_scores = {d[1] for d in fs_docs_with_scores}
    # Always append 0 as the lower bound
    fs_doc_distinct_scores = list(fs_doc_distinct_scores) + [0.0]
    fs_doc_distinct_scores.sort(reverse=True)
    fs_doc_id2score_lower_bound = {}
    for d, s in fs_doc_id2upper_bound.items():
        for s2_idx, s2 in enumerate(fs_doc_distinct_scores):
            if s == s2:
                fs_doc_id2score_lower_bound[d] = fs_doc_distinct_scores[s2_idx + 1]

    return fs_doc_id2upper_bound, fs_doc_id2score_lower_bound


for bench in tqdm(BENCHMARKS):
    # set the current benchmark for all rankers
    for ranker in RANKING_STRATEGIES:
        ranker.set_benchmark(bench)
    corpus_collections = [c for c in bench.document_collections]
   # if "PubMed" not in corpus_collections:
   #     corpus_collections.append("PubMed")
    corpus = DocumentCorpus(collections=corpus_collections)
    retriever = DocumentRetriever()
    print('==' * 60)
    print('==' * 60)
    print(f'Running benchmark: {bench}')
    print('==' * 60)
    print('==' * 60)
    for first_stage in FIRST_STAGE_NAMES:
        print('--' * 60)
        print(f'Running with first stage: {first_stage}')
        print('--' * 60)
        for concept_strategy in CONCEPT_STRATEGIES:
            prefix = first_stage + '_' + concept_strategy
            stats_dir = os.path.join(RESULT_DIR, 'statistics')
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            result_dir = os.path.join(RESULT_DIR, prefix)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            statistics_data = []
            topic2ids = {}
            if first_stage != ["GivenByBenchmark"]:
                path = os.path.join(RESULT_DIR_FIRST_STAGE, f'{bench.name}_{first_stage}_{concept_strategy}.txt')
                topic2ids = load_document_ids_from_runfile(path)

            ranker2result_lines = {}
            for idx, q in enumerate(bench.topics):
                print('--' * 60)
                print(f'Evaluating query {q}')
                print(f'Query components: {list(q.get_query_components())}')
                analyzed_query = AnalyzedQuery(q, concept_strategy=concept_strategy)
                analyzed_query_statistics = analyzed_query.get_statistics()

                if first_stage == "GivenByBenchmark":
                    # every document has the perfect score
                    fs_docs_with_scores = [(d, 1.0) for d in bench.topic_id2docs[q.query_id]]
                else:
                    if int(q.query_id) in topic2ids:
                        fs_docs_with_scores = topic2ids[int(q.query_id)]
                    else:
                        fs_docs_with_scores = list()

                fs_doc_ids = [d[0] for d in fs_docs_with_scores]
                # Get lower and upper bound of document ids scores
                fs_doc_id2upper_bound, fs_doc_id2lower_bound = compute_first_stage_scores(fs_docs_with_scores)

                print(f'Querying for {len(fs_docs_with_scores)} documents...')
                narrative_docs = list(retriever.retrieve_narrative_documents_for_collections(fs_doc_ids,
                                                                                             bench.document_collections))
                narrative_docs = sorted(narrative_docs, key=lambda x: x.document_id_source)
                for d in narrative_docs:
                    d.prepare_with_min_confidence()

                analyzed_query_statistics.update(analyzed_query.get_document_statistics(narrative_docs))
                statistics_data.append(analyzed_query_statistics)

                # compute matching fragments
                gf_path = os.path.join(RESULT_DIR_FIRST_STAGE,
                                       f'{bench.name}_{first_stage}_{concept_strategy}_pos2prov.json')
                gf = GraphFragment(gf_path)
                fragments = list(gf.matches(analyzed_query, doc) for doc in narrative_docs)

                print(f'{len(narrative_docs)} documents retrieved')

                if len(narrative_docs) > 0:
                    print('--' * 60)
                    for ranker in RANKING_STRATEGIES:
                        if ranker.name not in ranker2result_lines:
                            ranker2result_lines[ranker.name] = list()
                        start = datetime.now()
                        result_lines = []
                        ranked_docs = ranker.rank_documents(analyzed_query, narrative_docs, corpus, fragments)
                        ranked_docs_adjusted = []
                        if len(ranked_docs) > 0:
                            for rank, (doc_id, score) in enumerate(ranked_docs):
                                fs_max = fs_doc_id2upper_bound[str(doc_id)]
                                # to differentiate between different intervalls, add 0.01
                                fs_min = fs_doc_id2lower_bound[str(doc_id)] + 0.01
                                fs_range = fs_max - fs_min

                                if fs_min > fs_max or fs_min < 0.0 or fs_max > 1.0 or fs_range < 0.0 or fs_range > 1.0:
                                    raise ValueError(
                                        f'First stage score computation yielded an error:{fs_min} {fs_max} {fs_range}')

                                norm_score = fs_min + score * fs_range
                                if norm_score > 1.0 or norm_score < 0.0:
                                    raise ValueError(
                                        f'Document {doc_id} received a score not in [0, 1] (score = {norm_score} / ranker = {ranker.name})')

                                ranked_docs_adjusted.append((doc_id, norm_score))

                            ranked_docs_adjusted.sort(key=lambda x: (x[1], x[0]), reverse=True)
                            for rank, (doc_id, norm_score) in enumerate(ranked_docs_adjusted):
                                result_line = f'{q.query_id}\tQ0\t{doc_id}\t{rank + 1}\t{norm_score}\t{ranker.name}'
                                result_lines.append(result_line)

                        time_taken = datetime.now() - start
                        print(f'{time_taken}s to compute {ranker.name}')

                        ranker2result_lines[ranker.name].extend(result_lines)

            print('--' * 60)
            print('Writing result files...')
            for r in RANKING_STRATEGIES:
                results = ranker2result_lines[r.name]
                path = os.path.join(result_dir, f'{bench.name}_{r.name}.txt')
                with open(path, 'wt') as f:
                    f.write('\n'.join(results))

            # Execute the weighted ranker
            print('--' * 60)
            print('Executing weighted rankers...')
            run_weighted_ranker(benchmark=bench, weight_matrix=WEIGHT_MATRIX, ranker_bases=RANKER_BASES,
                                strategy=concept_strategy, first_stage=first_stage)

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
