from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class TfIdfAvgDocumentRanker(BaseDocumentRanker):
    def __init__(self, name="TfIdfAvgDocumentRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        scores = list()
        for spo in fragment:
            scores.append(BaseDocumentRanker.get_tf_idf(statement=spo, doc=doc, corpus=corpus))
        return sum(scores) / len(scores)
