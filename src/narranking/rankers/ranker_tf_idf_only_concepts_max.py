from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class TfIdfOnlyConceptsMaxDocumentRanker(BaseDocumentRanker):
    def __init__(self, name="TfIdfOnlyConceptsMaxDocumentRanker"):
        super().__init__(name=name)

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        scores = list()
        for spo in fragment:
            subject_tfidf = BaseDocumentRanker.get_concept_tf_idf(spo[0], doc=doc, corpus=corpus)
            object_tfidf = BaseDocumentRanker.get_concept_tf_idf(spo[2], doc=doc, corpus=corpus)
            tfidf = (subject_tfidf + object_tfidf) / 2.0
            scores.append(tfidf)

        return max(scores)
