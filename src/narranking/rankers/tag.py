from typing import List, Set

from narranking.corpus import DocumentCorpus
from narranking.document import AnalyzedNarrativeDocument
from narranking.query import AnalyzedQuery
from narranking.rankers.abstract import AbstractDocumentRanker, AbstractDocumentRankerWeighted


class TagFrequencyRanker(AbstractDocumentRanker):
    def __init__(self):
        super().__init__(name="TagFrequencyRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []
        for doc in narrative_documents:
            concepts: Set[str] = doc.concepts.intersection(query.concepts)
            score: float = sum([doc.concept2frequency[c] for c in concepts if c in doc.concept2frequency])
            doc_scores.append((doc.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)


class TagFrequencyRankerWeighted(AbstractDocumentRankerWeighted):
    def __init__(self):
        super().__init__(name="TagFrequencyRankerWeighted")

    def rank_query_part(self, doc: AnalyzedNarrativeDocument, concepts: Set[str]):
        concepts: Set[str] = doc.concepts.intersection(concepts)
        return sum([doc.concept2frequency[c] for c in concepts if c in doc.concept2frequency])
