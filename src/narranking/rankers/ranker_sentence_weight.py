from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class SentenceWeightRanker(BaseDocumentRanker):
    def __init__(self, name="SentenceWeightRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):

        fragment_weights = []
        for statement in fragment:
            statement_weights = []
            for sentence in doc.spo2sentences[statement]:
                statement_weights.append(1.0 / len(doc.sentence2spo[sentence]))

            fragment_weights.append(sum(statement_weights) / len(statement_weights))

        return sum(fragment_weights) / len(fragment_weights)
