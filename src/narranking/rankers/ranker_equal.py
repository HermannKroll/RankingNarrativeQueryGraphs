from typing import List

from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class EqualDocumentRanker(BaseDocumentRanker):

    def __init__(self):
        super().__init__(name="EqualDocumentRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus, fragments: list):
        return list([(d.document_id_source, 1.0) for d in narrative_documents])
