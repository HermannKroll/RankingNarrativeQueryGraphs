from tqdm import tqdm

from narraplay.documentranking.config import MINIMUM_TRANSLATION_THRESHOLD
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.run_config import BENCHMARKS, MINIMUM_COMPONENTS_IN_QUERY


def main() -> int:
    for bench in tqdm(BENCHMARKS):
        print(f'\n\n{bench.name}')
        query_scores = []
        for idx, q in enumerate(bench.topics):
            print('--' * 60)
            print(f'Evaluating query {q}')
            print(f'Query components: {list(q.get_query_components())}')
            if len(list(q.get_query_components())) < MINIMUM_COMPONENTS_IN_QUERY:
                print(f'Skipping topic {q.query_id} (less than {MINIMUM_COMPONENTS_IN_QUERY} components)')
                continue
            analyzed_query = AnalyzedQuery(q, concept_strategy="likesimilarity")
            score = analyzed_query.get_query_translation_score()
            for component, entities in analyzed_query.component2concepts.items():
                print(f'\t{component}\t--->\t{entities}')
                if not entities:
                    continue
                max_score = max([analyzed_query.concept2score[e] for e in entities])
                min_score = min([analyzed_query.concept2score[e] for e in entities])
                avg_score = sum([analyzed_query.concept2score[e] for e in entities]) / len(entities)
                print(f'\t Translation score: max: {max_score} / min: {min_score} / avg: {avg_score}')

            if score < MINIMUM_TRANSLATION_THRESHOLD:
                print(f'Skipping topic {q.query_id} (translation score {score} < {MINIMUM_TRANSLATION_THRESHOLD})')
                continue
            print()

            query_scores.append((q.query_id, score))

        print('--' * 60)
        print('Results for benchmark            : ' + bench.name)
        print('No. of queries                   : ' + str(len(bench.topics)))
        print('No. of queries with score >= 0.9 : ' + str(len(query_scores)))
        print('Query translation scores         : ' + str(query_scores))

    return 0


if __name__ == '__main__':
    main()
