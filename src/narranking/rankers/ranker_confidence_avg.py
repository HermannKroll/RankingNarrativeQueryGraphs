from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class ConfidenceAvgDocumentRanker(BaseDocumentRanker):
    def __init__(self, name="ConfidenceAvgDocumentRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        scores = list()
        for s, p, o in fragment:
            scores.append(sum(doc.spo2confidences[(s, p, o)]) / len(doc.spo2confidences[(s, p, o)]))
        return min(scores)
