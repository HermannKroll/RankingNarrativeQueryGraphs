from tqdm import tqdm

from narranking.benchmark import Benchmark
from narranking.query import AnalyzedQuery

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


def main() -> int:
    for bench in tqdm(BENCHMARKS):
        print(f'\n\n{bench.name}')
        for idx, q in enumerate(bench.topics):
            print('--' * 60)
            print(f'Evaluating query {q}')
            print(f'Query components: {list(q.get_query_components())}')
            analyzed_query = AnalyzedQuery(q, concept_strategy="hybrid")
            if len(analyzed_query.partial_concepts) > 1:
                print(f'Gene: {analyzed_query.partial_concepts[1]} ({len(list(q.get_query_components())[1])} genes)')
            else:
                print(f'Gene: no upper case letters')
            print(f'Query String (no gene mod): {q.get_benchmark_string(gene_modified=False, demographic=False)}')
            print(f'Query String (with gene mod): {q.get_benchmark_string(gene_modified=True, demographic=False)}')

    return 0


if __name__ == '__main__':
    main()
