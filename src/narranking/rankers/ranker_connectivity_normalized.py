from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker
from narraplay.documentranking.rankers.ranker_connectivity import ConnectivityDocumentRanker


class ConnectivityNormalizedDocumentRanker(ConnectivityDocumentRanker):
    def __init__(self):
        super().__init__(name="ConnectivityNormalizedDocumentRanker")

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        return super().rank_document_fragment(query, doc, corpus, fragment) / len(doc.graph)
