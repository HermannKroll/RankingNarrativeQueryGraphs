import pyterrier as pt
from tqdm import tqdm

from narraplay.documentranking.create_document_index import BenchmarkIndex
from narraplay.documentranking.run_config import BENCHMARKS


def main() -> int:
    if not pt.started():
        pt.init()
    for benchmark in tqdm(BENCHMARKS):
        index = BenchmarkIndex(benchmark)
        index.perform_bm25_retrieval()

    return 0


if __name__ == '__main__':
    main()
