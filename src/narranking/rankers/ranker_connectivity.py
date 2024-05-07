from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class ConnectivityDocumentRanker(BaseDocumentRanker):
    def __init__(self, name="ConnectivityDocumentRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        nodes = set()
        for s, _, o in fragment:
            nodes.add(s)
            nodes.add(o)

        counter = 0
        for s, _, o in doc.graph:
            # don't count edges between the fragment
            if s in nodes and o in nodes:
                continue
            if s in nodes or o in nodes:
                counter += 1

        return counter
