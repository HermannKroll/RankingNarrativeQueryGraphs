from typing import List, Tuple

from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_b25_base import BM25ReRankerBase


class BM25Text(BM25ReRankerBase):
    def __init__(self, name="BM25Text"):
        super().__init__(name=name)

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus, fragments: list) -> List[Tuple[str, float]]:
        query_str = self.filter_query_string(query.topic.get_benchmark_string())
        return self.score_with_bm25(query_str, narrative_documents)
