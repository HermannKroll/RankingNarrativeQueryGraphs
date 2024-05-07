import os

import pandas as pd
import pyterrier as pt
from tqdm import tqdm

from kgextractiontoolbox.backend.database import Session
from kgextractiontoolbox.backend.models import Document
from kgextractiontoolbox.backend.retrieve import iterate_over_all_documents_in_collection
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import PYTERRIER_INDEX_PATH, \
    BM25_RANKED_DOCUMENT_CUTOFF, RESULT_DIR_FIRST_STAGE, RESULT_DIR_BASELINES
from narraplay.documentranking.run_config import BENCHMARKS
from narraplay.documentranking.translator import DocumentTranslator


class BenchmarkIndex:

    def __init__(self, benchmark: Benchmark):
        self.benchmark = benchmark
        self.name = benchmark.name
        self.path = os.path.join(PYTERRIER_INDEX_PATH, self.name)
        self.index = None
        if os.path.isdir(self.path):
            print(f'Loading index from {self.path}')
            self.index = pt.IndexFactory.of(self.path, memory=True)
        else:
            self.create_index()

    def create_index(self, stemmer="EnglishSnowballStemmer", stopwords="terrier"):
        values = []
        session = Session.get()
        translator = DocumentTranslator()
        print(f'Create index for {self.name} (collections = {self.benchmark.document_collections})')
        for collection in self.benchmark.document_collections:
            print(f'\nIterating over all documents in {collection}')
            total = session.query(Document).filter(Document.collection == collection).count()
            known_ids = set()
            # we focus on titles and abstract if not otherwise specified by benchmark
            consider_sections = self.benchmark.has_fulltexts

            # iterate over all documents for that collection
            for d in tqdm(
                    iterate_over_all_documents_in_collection(session, collection, consider_sections=consider_sections),
                    total=total):

                doc_id_source = translator.translate_document_id_art2source(d.id, collection)
                # Filter on PubMed documents
                if collection == 'PubMed' and self.benchmark.get_documents_for_baseline():
                    if int(doc_id_source) not in self.benchmark.get_documents_for_baseline():
                        # Skip documents that are not relevant for that baseline
                        continue

                # ensure that each source id is unique
                assert doc_id_source not in known_ids
                known_ids.add(doc_id_source)

                text = d.get_text_content(sections=consider_sections)
                if text.strip():
                    values.append([str(doc_id_source), text])
        print()
        print(f'{len(values)} documents retrieved...')
        print('Creating dataframe...')
        df = pd.DataFrame(values, columns=['docno', 'text'])
        print('Creating index...')
        pd_indexer = pt.DFIndexer(self.path, verbose=True, overwrite=True, stemmer=stemmer, stopwords=stopwords)
        self.index = pd_indexer.index(df["text"], df["docno"])
        print('Finished!')

    def perform_bm25_retrieval(self):
        print(f'Performing BM25 Retrieval on {self.name}...')
        bm25 = pt.BatchRetrieve(self.index, wmodel="BM25")
        result_lines = []
        for topic in tqdm(self.benchmark.topics):
            rtr = bm25.search("".join([x if x.isalnum() else " " for x in topic.get_benchmark_string()]))
            scored_docs = []
            for index, row in rtr.iterrows():
                scored_docs.append((row["docno"], row["score"]))
            scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            scored_docs = scored_docs[:BM25_RANKED_DOCUMENT_CUTOFF]

            max_score = scored_docs[0][1]
            rank = 0
            for doc, score in scored_docs:
                norm_score = score / max_score
                result_lines.append(f'{topic.query_id}\tQ0\t{doc}\t{rank + 1}\t{norm_score}\tBM25')
                rank += 1

        result_file_path = os.path.join(RESULT_DIR_BASELINES, f'{self.name}_BM25.txt')
        print(f'Writing results to {result_file_path}')
        with open(result_file_path, 'wt') as f:
            f.write('\n'.join([l for l in result_lines]))
        print('Finished')


def main() -> int:
    if not pt.started():
        pt.init()
    for benchmark in tqdm(BENCHMARKS):
        index = BenchmarkIndex(benchmark)


if __name__ == '__main__':
    main()
