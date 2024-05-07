import itertools

from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class RelationalSimTranslationDocumentRanker(BaseDocumentRanker):
    def __init__(self):
        super().__init__(name="RelationalSimTranslationDocumentRanker")

    def rank_document_fragment(self, query: AnalyzedQuery, doc: AnalyzedNarrativeDocument,
                               corpus: DocumentCorpus, fragment: list):
        scores = list()
        for spo in fragment:
            confidence = max(doc.spo2confidences[spo])

            visited = set()
            for statement in itertools.chain(doc.concept2statement[spo[0]], doc.concept2statement[spo[2]]):
                # iterate over each edge once
                n_spo = (statement.subject_id, statement.relation, statement.object_id)
                if n_spo in visited:
                    continue
                visited.add(n_spo)

                # skip edges between the fragment
                if n_spo[0] == spo[0] and n_spo[2] == spo[2]:
                    continue

                # neighbour edge = edge that is connected to the fragment via subject or object
                if n_spo[0] == spo[0] or n_spo[2] == spo[2]:
                    if n_spo[0] == spo[0]:
                        translation_score = query.concept2score[spo[0]]
                    else:
                        translation_score = query.concept2score[spo[2]]

                    tf_idf = BaseDocumentRanker.get_tf_idf(statement=n_spo, doc=doc, corpus=corpus)

                    score = translation_score * confidence * tf_idf
                    scores.append(score)

        # we might do not have neighbour edges
        if len(scores) == 0:
            return 0.0

        return sum(scores) / len(scores)
