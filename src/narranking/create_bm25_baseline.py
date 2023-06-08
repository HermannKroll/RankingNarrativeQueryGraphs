import os.path

import pyterrier as pt
import pandas as pd
from tqdm import tqdm

from kgextractiontoolbox.backend.database import Session
from kgextractiontoolbox.backend.models import Document
from narranking.benchmark import Benchmark
from narranking.config import PYTERRIER_INDEX_PATH, RANKED_DOCUMENT_CUTOFF
from narranking.translator import DocumentTranslator

BENCHMARKS = [
    Benchmark("trec-pm-2017-cds", "clinicaltrials/2017/trec-pm-2017", ["trec-pm-2017"]),
    Benchmark("trec-pm-2018-cds", "clinicaltrials/2017/trec-pm-2018", ["trec-pm-2018-cds"]),
    Benchmark("trec-pm-2019-cds", "clinicaltrials/2019/trec-pm-2019", ["trec-pm-2019-cds"]),
    Benchmark("trec-pm-2017-abstracts", "medline/2017/trec-pm-2017", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2018-abstracts", "medline/2017/trec-pm-2018", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2019-abstracts", "clinicaltrials/2019/trec-pm-2019", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2020-abstracts", "", ["PubMed"], load_from_file=True)
]


class BenchmarkIndex:

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.name = benchmark.name
        self.path = os.path.join(PYTERRIER_INDEX_PATH, self.name)
        self.index = None
        if os.path.isdir(self.path):
            print(f'Loading index from {self.path}')
            self.index = pt.IndexFactory.of(self.path)
        else:
            self.create_index()

    def create_index(self):
        values = []
        session = Session.get()
        translator = DocumentTranslator()
        print(f'Create index for {self.name} (collections = {self.benchmark.document_collections})')
        for collection in self.benchmark.document_collections:
            print(f'\nIterating over all documents in {collection}')

            total = session.query(Document).filter(Document.collection == collection).count()
            doc_query = session.query(Document).filter(Document.collection == collection)
            for d in tqdm(doc_query, total=total):
                doc_id_source = translator.translate_document_id_art2source(d.id, collection)
                # Filter on PubMed documents
                if collection == 'PubMed' and self.benchmark.get_documents_for_baseline():
                    if int(doc_id_source) not in self.benchmark.get_documents_for_baseline():
                        # Skip documents that are not relevant for that baseline
                        continue

                text = f'{d.title} {d.abstract}'
                if text.strip():
                    values.append([str(doc_id_source), text])
        print()
        print(f'{len(values)} documents retrieved...')
        print('Creating dataframe...')
        df = pd.DataFrame(values, columns=['docno', 'text'])
        print('Creating index...')
        pd_indexer = pt.DFIndexer(self.path, verbose=True)
        self.index = pd_indexer.index(df["text"], df["docno"])
        print('Finished!')

    def perform_bm25_retrieval(self, demographic, gene_modified):
        print(f'Performing BM25 Retrieval on {self.name}...')
        bm25 = pt.BatchRetrieve(self.index, wmodel="BM25")
        result_lines = []
        for topic in self.benchmark.topics:
            rtr = bm25.search(topic.get_benchmark_string(gene_modified=gene_modified, demographic=demographic))
            scored_docs = []
            for index, row in rtr.iterrows():
                scored_docs.append((row["docno"], row["score"]))
            scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            scored_docs = scored_docs[:RANKED_DOCUMENT_CUTOFF]

            max_score = scored_docs[0][1]
            rank = 0
            for doc, score in scored_docs:
                norm_score = score / max_score
                result_lines.append(f'{topic.query_id}\tQ0\t{doc}\t{rank + 1}\t{norm_score}\tBM25')
                rank += 1

        demographic_str = "without_demographic"
        if demographic:
            demographic_str = "with_demographic"

        gene = "gene"
        if gene_modified:
            gene = "modified_gene"

        result_file_path = os.path.join(PYTERRIER_INDEX_PATH,
                                        f'{self.name}_FirstStageBM25Own_{demographic_str}_{gene}.txt')
        print(f'Writing results to {result_file_path}')
        with open(result_file_path, 'wt') as f:
            f.write('\n'.join([l for l in result_lines]))
        print('Finished')


def main() -> int:
    pt.init()
    for benchmark in tqdm(BENCHMARKS):
        index = BenchmarkIndex(benchmark)

        for demographic in [True, False]:
            for gene_modified in [True, False]:
                index.perform_bm25_retrieval(demographic=demographic, gene_modified=gene_modified)

    return 0


if __name__ == '__main__':
    main()
