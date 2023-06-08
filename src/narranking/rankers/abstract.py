from typing import List, Set

from narranking.corpus import DocumentCorpus
from narranking.document import AnalyzedNarrativeDocument
from narranking.query import AnalyzedQuery


class AbstractDocumentRanker:

    def __init__(self, name):
        self.name = name

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        pass


class AbstractDocumentRankerWeighted(AbstractDocumentRanker):

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        """
        e.g.:
        DOCUMENTS D1, D2, CONCEPTS A, B, C, WEIGHT = 1 / #CONCEPTS = 0.33

        Concept scores (metric dependent)
        D1_A = 1000, D1_B = 10, D1_C = 10
        D2_A =   10, D2_B = 10, D2_C = 10

        Concept maximum
        A_max = D1_A, B_max = D1_B, C_max = D1_C

        Document scores
        D1_score = (D1_A / A_max + D1_B / B_max + D1_C / C_max) * WEIGHT = 1
        D2_score = (D2_A / A_max + D2_B / B_max + D2_C / C_max) * WEIGHT = 0.66
        """
        max_scores = self.max_scores(narrative_documents, query.partial_concepts)
        doc_scores = [(d.document_id_source, sum(
            [self.rank_query_part(d, c) / ms for (c, ms) in zip(query.partial_concepts, max_scores) if ms != 0]
        ) * query.partial_weight) for d in narrative_documents]
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    def max_scores(self, narrative_documents: List[AnalyzedNarrativeDocument], partial_concepts: List[Set]):
        return [max([self.rank_query_part(d, c) for d in narrative_documents]) for c in partial_concepts]

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        raise NotImplementedError()
