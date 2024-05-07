from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class IDFMinDocumentRanker(BaseDocumentRanker):
    def __init__(self, name="IDFMinDocumentRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        scores = list()
        for spo in fragment:
            scores.append(corpus.get_idf_score(spo))
        return min(scores)
