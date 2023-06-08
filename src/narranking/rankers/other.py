from typing import List, Set

import networkx

from narranking.corpus import DocumentCorpus
from narranking.document import AnalyzedNarrativeDocument
from narranking.query import AnalyzedQuery
from narranking.rankers.abstract import AbstractDocumentRanker, AbstractDocumentRankerWeighted
from narranking.rankers.graph import PathFrequencyRanker


class AdjacentEdgesRanker(PathFrequencyRanker):
    def __init__(self, name="AdjacentEdgesRanker"):
        super().__init__(name=name)

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for doc in narrative_documents:
            graph = self._document_graph(doc)
            score = self._evaluate_confidence_score(query, doc, graph)
            doc_scores.append((doc.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _evaluate_confidence_score(query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                                   graph: networkx.MultiGraph) -> float:
        return sum((len(graph.edges(c)) for c in query.concepts))


class AdjacentEdgesRankerWeighted(PathFrequencyRanker):
    def __init__(self, name="AdjacentEdgesRankerWeighted"):
        super().__init__(name=name)

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        max_scores = [max([self.rank_query_part(d, c) for d in narrative_documents]) for c in query.partial_concepts]
        doc_scores = [(d.document_id_source, sum(
            [self.rank_query_part(d, c) / ms for (c, ms) in zip(query.partial_concepts, max_scores) if ms != 0]
        ) * query.partial_weight) for d in narrative_documents]
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]) -> float:
        graph: networkx.MultiGraph = self._document_graph(doc)
        return sum((len(graph.edges(c)) for c in concepts))


class ConfidenceAdjacentEdgesRanker(AdjacentEdgesRanker):
    def __init__(self):
        super().__init__(name="ConfidenceAdjacentEdgesRanker")

    @staticmethod
    def _evaluate_confidence_score(query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                                   graph: networkx.MultiGraph) -> float:
        score: float = 0
        for concept in query.concepts:
            edges = graph.edges(concept)
            for s, o in edges:
                confidences = [stmt.confidence for stmt in doc.so2statement[(s, o)]]
                score += sum(confidences)
        return score


class ConfidenceAdjacentEdgesRankerWeighted(AdjacentEdgesRankerWeighted):
    def __init__(self):
        super().__init__(name="ConfidenceAdjacentEdgesRankerWeighted")

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]) -> float:
        score: float = 0
        graph: networkx.MultiGraph = self._document_graph(doc)
        for concept in concepts:
            edges = graph.edges(concept)
            for s, o in edges:
                confidences = [stmt.confidence for stmt in doc.so2statement[(s, o)]]
                score += sum(confidences)
        return score


class BM25Tag(AbstractDocumentRanker):

    def __init__(self):
        self.k = 2.0
        self.b = 0.75
        super().__init__(name="BM25Tag")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for doc in narrative_documents:
            score = 0
            for c in query.concepts:
                a = corpus.idf_concept(c) * (doc.get_concept_frequency(c) * (self.k + 1))
                b = doc.get_concept_frequency(c) + self.k * (1 - self.b + self.b * doc.get_length_in_concepts())
                score += a / b
            doc_scores.append((doc.document_id_source, score))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class BM25TagWeighted(AbstractDocumentRankerWeighted):
    def __init__(self):
        self.k = 2.0
        self.b = 0.75
        self.corpus = None
        super().__init__(name="BM25TagWeighted")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        self.corpus = corpus
        doc_scores = []
        max_scores = self.max_scores(narrative_documents, query.partial_concepts)
        for doc in narrative_documents:
            score = 0
            for max_score, concepts in zip(max_scores, query.partial_concepts):
                if max_score == 0:
                    continue
                im_score = 0
                for c in concepts:
                    a = self.corpus.idf_concept(c) * (doc.get_concept_frequency(c) * (self.k + 1))
                    b = doc.get_concept_frequency(c) + self.k * (1 - self.b + self.b * doc.get_length_in_concepts())
                    im_score += a / b
                score += im_score / max_score
            score *= query.partial_weight
            doc_scores.append((doc.document_id_source, score))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        score = 0
        for c in concepts:
            a = self.corpus.idf_concept(c) * (doc.get_concept_frequency(c) * (self.k + 1))
            b = doc.get_concept_frequency(c) + self.k * (1 - self.b + self.b * doc.get_length_in_concepts())
            score += a / b
        return score
