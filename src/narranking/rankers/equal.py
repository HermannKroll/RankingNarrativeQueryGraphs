from typing import List

from narranking.corpus import DocumentCorpus
from narranking.document import AnalyzedNarrativeDocument
from narranking.query import AnalyzedQuery
from narranking.rankers.abstract import AbstractDocumentRanker


class EqualDocumentRanker(AbstractDocumentRanker):

    def __init__(self):
        super().__init__(name="EqualDocumentRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        return list([(d.document_id_source, 2.0) for d in narrative_documents])
