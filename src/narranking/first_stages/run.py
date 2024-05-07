import json
import logging
import os
from collections import defaultdict

from tqdm import tqdm

from narraplay.documentranking.config import RESULT_DIR_FIRST_STAGE, MINIMUM_TRANSLATION_THRESHOLD
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.run_config import CONCEPT_STRATEGIES, BENCHMARKS, FIRST_STAGES, FIRST_STAGE_CUTOFF, \
    MINIMUM_COMPONENTS_IN_QUERY


def main():
    for stage in FIRST_STAGES:
        for bench in tqdm(BENCHMARKS):
            print('==' * 60)
            print('==' * 60)
            print(f'Running configuration')
            print(f'- FIRST STAGE: {stage.name}')
            print(f'- BENCHMARK: {bench}')
            print('==' * 60)
            print('==' * 60)

            for concept_strategy in CONCEPT_STRATEGIES:
                result_dir = RESULT_DIR_FIRST_STAGE
                stats_dir = os.path.join(result_dir, 'statistics')
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)

                statistics_data = {}
                result_lines = []
                pos2prov_collection_dict = defaultdict(dict)
                for idx, q in enumerate(bench.topics):
                    print('--' * 60)
                    print(f'Evaluating query {q}')
                    print(f'Query components: {list(q.get_query_components())}')
                    analyzed_query = AnalyzedQuery(q, concept_strategy=concept_strategy)

                    if len(analyzed_query.component2concepts) < MINIMUM_COMPONENTS_IN_QUERY:
                        statistics_data[q.query_id] = {"skipped": f"less than {MINIMUM_COMPONENTS_IN_QUERY} concepts",
                                                       "no_concepts": len(analyzed_query.component2concepts)}
                        print(f'Skipping topic {q.query_id} (less than {MINIMUM_COMPONENTS_IN_QUERY} concepts)')
                        continue

                    trans_score = analyzed_query.get_query_translation_score()
                    statistics_data[q.query_id] = {"no_concepts": len(analyzed_query.component2concepts),
                                                   "translation_score": trans_score}
                    if trans_score < MINIMUM_TRANSLATION_THRESHOLD:
                        statistics_data[q.query_id].update(
                            {"skipped": f'translation score less than {MINIMUM_TRANSLATION_THRESHOLD}'})
                        print(
                            f'Skipping topic {q.query_id} (translation score {trans_score} < {MINIMUM_TRANSLATION_THRESHOLD})')
                        continue

                    ranked_docs = []
                    for collection in bench.document_collections:
                        # print(f'Querying {collection} documents with: {fs_query}')
                        docs_for_c, statistics, pos2prov_dict = stage.retrieve_documents(analyzed_query, collection,
                                                                                         bench)
                        pos2prov_collection_dict[collection][q.query_id] = pos2prov_dict
                        ranked_docs.extend(docs_for_c)
                        statistics_data[q.query_id].update(statistics)

                    print(f'Received {len(ranked_docs)} documents')
                    ranked_docs.sort(key=lambda x: (x[1], x[0]), reverse=True)
                    if FIRST_STAGE_CUTOFF > 0:
                        ranked_docs = ranked_docs[:FIRST_STAGE_CUTOFF]
                        print(f'First stage cutoff applied -> {len(ranked_docs)} remaining')
                    count = 0
                    for rank, (did, score) in enumerate(ranked_docs):
                        result_line = f'{q.query_id}\tQ0\t{did}\t{rank + 1}\t{score}\t{stage.name}'
                        result_lines.append(result_line)
                        count += 1

                path = os.path.join(result_dir, f'{bench.name}_{stage.name}_{concept_strategy}.txt')
                print(f'Write ranked results to {path}')
                with open(path, 'wt') as f:
                    f.write('\n'.join(result_lines))

                path = os.path.join(stats_dir, f'{bench.name}_{stage.name}_{concept_strategy}.json')
                print(f'Write statistics to {path}')
                with open(path, 'wt') as f:
                    json.dump(statistics_data, f, indent=2)
                print('--' * 60)

                path = os.path.join(result_dir, f'{bench.name}_{stage.name}_{concept_strategy}_pos2prov.json')
                print(f'Write pos2prov to {path}')
                with open(path, 'wt') as f:
                    json.dump(pos2prov_collection_dict, f, indent=2)
                print('--' * 60)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)

    main()
