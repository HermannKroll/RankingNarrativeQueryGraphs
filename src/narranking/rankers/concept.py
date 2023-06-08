from typing import List, Set

from narranking.corpus import DocumentCorpus
from narranking.document import AnalyzedNarrativeDocument
from narranking.query import AnalyzedQuery
from narranking.rankers.abstract import AbstractDocumentRanker, AbstractDocumentRankerWeighted


class ConceptDocumentRanker(AbstractDocumentRanker):

    def __init__(self):
        super().__init__(name="ConceptDocumentRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for d in narrative_documents:
            doc_scores.append((d.document_id_source, len(query.concepts.intersection(d.concepts))))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class ConceptDocumentRankerWeighted(AbstractDocumentRanker):

    def __init__(self):
        super().__init__(name="ConceptDocumentRankerWeighted")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for d in narrative_documents:
            score = 0
            for concepts in query.partial_concepts:
                num_matched = len(concepts.intersection(d.concepts))
                if len(concepts) > 0:
                    score += num_matched / len(concepts)
            score *= query.partial_weight
            doc_scores.append((d.document_id_source, score))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class ConceptFrequencyDocumentRanker(AbstractDocumentRanker):

    def __init__(self):
        super().__init__(name="ConceptFrequencyDocumentRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for d in narrative_documents:
            concept_frequency = sum([d.concept2frequency[c] for c in query.concepts if c in d.concept2frequency])
            doc_scores.append((d.document_id_source, concept_frequency))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class ConceptFrequencyDocumentRankerWeighted(AbstractDocumentRankerWeighted):

    def __init__(self):
        super().__init__(name="ConceptFrequencyDocumentRankerWeighted")

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        return sum([doc.concept2frequency[c] for c in concepts if c in doc.concept2frequency])
