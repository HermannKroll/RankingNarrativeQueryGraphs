import math
from typing import List, Tuple

from narraplay.documentranking.corpus import DocumentCorpus
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.rankers.ranker_base import BaseDocumentRanker


class DocLengthDocumentRanker(BaseDocumentRanker):
    def __init__(self):
        super().__init__(name="DocLengthDocumentRanker")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus, fragments: list) -> List[Tuple[str, float]]:
        max_length = max(len(d.graph) for d in narrative_documents)

        results = list()
        for doc, d_fragments in zip(narrative_documents, fragments):
            score = len(doc.graph) / max_length
            results.append((doc.document_id_source, score))
        results.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return results
