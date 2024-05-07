from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_connectivity import ConnectivityDocumentRanker


class ConnectivityInverseDocumentRanker(ConnectivityDocumentRanker):
    def __init__(self):
        super().__init__(name="ConnectivityInverseDocumentRanker")

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        no_connected_components = super().rank_document_fragment(query, doc, corpus, fragment)
        if no_connected_components == 0:
            return 0.0
        return 1.0 / no_connected_components
