import os
from typing import List

import pandas as pd
import pyterrier as pt

from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import PYTERRIER_INDEX_PATH
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class BM25ReRankerBase(BaseDocumentRanker):

    def __init__(self, name, termpipelines='Stopwords,PorterStemmer'):
        if not pt.started():
            pt.init()

        self.bm25pipeline = None
        self.termpipelines = termpipelines

        super().__init__(name=name)

    def filter_query_string(self, query):
        return "".join([x if x.isalnum() else " " for x in query])

    def set_benchmark(self, benchmark: Benchmark):
        index_path = os.path.join(PYTERRIER_INDEX_PATH, benchmark.name)
        print(f'Load BM25 index from: {index_path}')
        bm25_index = pt.IndexFactory.of(index_path, memory=True)

        self.bm25pipeline = pt.BatchRetrieve(
            bm25_index,
            wmodel='BM25',
            properties={'termpipelines': self.termpipelines}
        )

    def score_with_bm25(self, query: str, documents: List[AnalyzedNarrativeDocument], normalize=True):
        assert len(query.strip()) > 0
        assert len(documents) > 0

        d_texts = []
        for doc in documents:
            d_texts.append(["q0", query, doc.document_id_source])

        df = pd.DataFrame(d_texts, columns=["qid", "query", "docno"])
        rtr = self.bm25pipeline(df)

        scored_docs = []
        for index, row in rtr.iterrows():
            # transform document id back to internal representation, e.g. PubMed_123 -> 123
            scored_docs.append(((row["docno"]), max(float(row["score"]), 0.0)))

        if normalize:
            # apply normalization
            max_score = max(sd[1] for sd in scored_docs)
            if max_score > 0.0:
                scored_docs = [(sd[0], sd[1] / max_score) for sd in scored_docs]

        scored_docs = sorted(scored_docs, key=lambda x: (x[1], x[0]), reverse=True)
        return scored_docs
