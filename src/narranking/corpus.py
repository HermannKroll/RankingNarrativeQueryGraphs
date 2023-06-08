from numpy import log as ln

from narranking.document import AnalyzedNarrativeDocument


class DocumentCorpus:

    def __init__(self, documents: [AnalyzedNarrativeDocument]):
        self.documents = documents
        self.document_count = len(documents)
        self.concept2docs = {}
        for doc in documents:
            for c in doc.concepts:
                if c not in self.concept2docs:
                    self.concept2docs[c] = 0
                else:
                    self.concept2docs[c] += 1

    def get_document_count_for_concept(self, concept):
        if concept in self.concept2docs:
            return self.concept2docs[concept]
        return 0

    def idf_concept(self, concept: str):
        a = self.document_count - self.get_document_count_for_concept(concept) + 0.5
        b = self.get_document_count_for_concept(concept) + 0.5
        return ln((a / b) + 1)
