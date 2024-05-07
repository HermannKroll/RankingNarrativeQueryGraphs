import itertools

from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker
from narraplay.documentranking.rankers.ranker_relational_sim import RelationalSimDocumentRanker


class RelationalSimNormalizedDocumentRanker(RelationalSimDocumentRanker):
    def __init__(self):
        super().__init__(name="RelationalSimNormalizedDocumentRanker")

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        scores = RelationalSimDocumentRanker.get_relational_similarity_scores(doc, corpus, fragment)
        return sum(scores) / len(scores)
