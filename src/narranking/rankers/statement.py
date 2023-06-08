import itertools
from typing import List, Set, Tuple

from narranking.corpus import DocumentCorpus
from narranking.document import AnalyzedNarrativeDocument
from narranking.query import AnalyzedQuery
from narranking.rankers.abstract import AbstractDocumentRanker, AbstractDocumentRankerWeighted


class StatementPartialOverlapDocumentRanker(AbstractDocumentRanker):

    def __init__(self):
        super().__init__(name="StatementPartialOverlapDocumentRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for d in narrative_documents:
            score = len(query.concepts.intersection(d.subjects)) + len(query.concepts.intersection(d.objects))
            doc_scores.append((d.document_id_source, score))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class StatementPartialOverlapDocumentRankerWeighted(AbstractDocumentRankerWeighted):

    def __init__(self):
        super().__init__(name="StatementPartialOverlapDocumentRankerWeighted")

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        return len(concepts.intersection(doc.subjects)) + len(concepts.intersection(doc.objects))


class StatementPartialOverlapFrequencyDocumentRankerWeighted(AbstractDocumentRankerWeighted):

    def __init__(self):
        super().__init__(name="StatementPartialOverlapFrequencyDocumentRankerWeighted")

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        score = 0
        for c in concepts:
            if c in doc.concept2statement:
                score += len(doc.concept2statement[c])
        return score


class StatementPartialOverlapConfidenceDocumentRankerWeighted(AbstractDocumentRankerWeighted):

    def __init__(self):
        super().__init__(name="StatementPartialOverlapConfidenceDocumentRankerWeighted")

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        score = 0
        for c in concepts:
            if c in doc.concept2statement:
                for s in doc.concept2statement[c]:
                    score += s.confidence
        return score


class StatementOverlapDocumentRanker(AbstractDocumentRanker):

    def __init__(self):
        super().__init__(name="StatementOverlapDocumentRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        query_concept_pairs = set(itertools.product(query.concepts, query.concepts))
        for d in narrative_documents:
            score = float(len(d.statement_concepts.intersection(query_concept_pairs)))
            doc_scores.append((d.document_id_source, score))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class StatementOverlapDocumentRankerWeighted(AbstractDocumentRankerWeighted):
    def __init__(self):
        super().__init__(name="StatementOverlapDocumentRankerWeighted")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        max_scores = self.max_scores(narrative_documents, query.partial_concepts)
        partial_concept_pairs = list(itertools.combinations(query.partial_concepts, 2))
        partial_concept_pairs_idx = list(itertools.combinations([*range(len(query.partial_concepts))], 2))
        doc_scores = []
        for d in narrative_documents:
            score = 0
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = set(itertools.product(c1, c2))
                pair_score = self.rank_query_part(d, query_concept_pairs)
                im_score = 0
                if max_scores[i1] != 0:
                    im_score += (pair_score / max_scores[i1])
                if max_scores[i2] != 0:
                    im_score += (pair_score / max_scores[i2])
                score += im_score / 2
            score *= 1.0 / len(partial_concept_pairs)
            doc_scores.append((d.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    def max_scores(self, narrative_documents: List[AnalyzedNarrativeDocument], partial_concepts: List[Set]):
        partial_concept_pairs = list(itertools.combinations(partial_concepts, 2))
        partial_concept_pairs_idx = list(itertools.combinations([*range(len(partial_concepts))], 2))
        max_scores = [0.0] * len(partial_concepts)
        for d in narrative_documents:
            # find the maximum for each partial concept itself
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = set(itertools.product(c1, c2))
                pair_score = self.rank_query_part(d, query_concept_pairs)
                max_scores[i1] = max(max_scores[i1], pair_score)
                max_scores[i2] = max(max_scores[i2], pair_score)
        return max_scores

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concept_pairs: Set[Tuple[str, str]]):
        return float(len(doc.statement_concepts.intersection(concept_pairs)))


class StatementOverlapFrequencyDocumentRankerWeighted(StatementOverlapDocumentRankerWeighted):
    def __init__(self):
        super().__init__()
        self.name = "StatementOverlapFrequencyDocumentRankerWeighted"

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concept_pairs: Set[Tuple[str, str]]):
        score = 0
        for pair in concept_pairs:
            if pair in doc.so2statement:
                score += len(doc.so2statement[pair])
        return score


class StatementOverlapConfidenceDocumentRankerWeighted(StatementOverlapDocumentRankerWeighted):
    def __init__(self):
        super().__init__()
        self.name = "StatementOverlapConfidenceDocumentRankerWeighted"

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concept_pairs: Set[Tuple[str, str]]):
        score = 0
        for pair in concept_pairs:
            if pair in doc.so2statement:
                for s in doc.so2statement[pair]:
                    score += s.confidence
        return score


class StatementFrequencyRanker(AbstractDocumentRanker):
    def __init__(self, name="StatementFrequencyRanker"):
        super().__init__(name=name)

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for doc in narrative_documents:
            score = len(self._relevant_statements(query, doc))
            doc_scores.append((doc.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _relevant_statements(query: AnalyzedQuery, narrative_document: AnalyzedNarrativeDocument):
        rev_stmts = set()
        for c in query.concepts:
            rev_stmts.update(narrative_document.concept2statement[c])
        return rev_stmts


class StatementFrequencyRankerWeighted(AbstractDocumentRankerWeighted):
    def __init__(self, name="StatementFrequencyRankerWeighted"):
        super().__init__(name=name)

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        return len(self._relevant_statements(concepts, doc))

    @staticmethod
    def _relevant_statements(concepts: set, narrative_document: AnalyzedNarrativeDocument):
        rev_stmts = set()
        for c in concepts:
            rev_stmts.update(narrative_document.concept2statement[c])
        return rev_stmts


class ConfidenceStatementFrequencyRanker(StatementFrequencyRanker):
    def __init__(self):
        super().__init__(name="ConfidenceStatementFrequencyRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for doc in narrative_documents:
            score = sum([s.confidence for s in self._relevant_statements(query, doc)])
            doc_scores.append((doc.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class ConfidenceStatementFrequencyRankerWeighted(StatementFrequencyRankerWeighted):
    def __init__(self):
        super().__init__(name="ConfidenceStatementFrequencyRankerWeighted")

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        return sum([s.confidence for s in self._relevant_statements(concepts, doc)])
