from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class ConfidenceMaxDocumentRanker(BaseDocumentRanker):
    def __init__(self, name="ConfidenceMaxDocumentRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        scores = list()
        for s, p, o in fragment:
            scores.append(max(doc.spo2confidences[(s, p, o)]))
        return max(scores)
