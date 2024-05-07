from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class ConceptCoverageDocumentRanker(BaseDocumentRanker):
    def __init__(self, name="ConceptCoverageDocumentRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        concepts = set()
        for s, p, o in fragment:
            concepts.add(s)
            concepts.add(o)

        scores = [doc.get_concept_coverage(c) for c in concepts]
        return min(scores)
