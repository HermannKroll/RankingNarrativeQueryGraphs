from typing import List

from kgextractiontoolbox.backend.models import DocumentTranslation
from kgextractiontoolbox.backend.database import Session


class DocumentTranslator:

    def __init__(self):
        self.__doc_translation_source2art = {}
        self.__doc_translation_art2source = {}

    def __cache_document_translation(self, document_collection: str):
        if document_collection not in self.__doc_translation_source2art:
            session = Session.get()
            query = session.query(DocumentTranslation.document_id, DocumentTranslation.source_doc_id)
            query = query.filter(DocumentTranslation.document_collection == document_collection)

            source2art = {}
            art2source = {}
            for r in query:
                # Check whether mapping is unique (1:1 mapping between source and db ids)
                assert r.document_id not in art2source
                assert r.source_doc_id not in source2art

                art2source[r.document_id] = r.source_doc_id
                source2art[r.source_doc_id] = r.document_id

            self.__doc_translation_source2art[document_collection] = source2art
            self.__doc_translation_art2source[document_collection] = art2source

    def translate_document_ids_art2source(self, document_ids: [int], document_collection: str) -> List[str]:
        # Hack for PubMed
        if document_collection == "PubMed":
            return [str(d) for d in document_ids]
        self.__cache_document_translation(document_collection)
        art2source = self.__doc_translation_art2source[document_collection]
        return [art2source[int(d)] for d in document_ids]

    def translate_document_id_art2source(self, document_id: int, document_collection: str) -> str:
        # Hack for PubMed
        if document_collection == "PubMed":
            return str(document_id)
        return self.translate_document_ids_art2source([document_id], document_collection)[0]

    def translate_document_ids_source2art(self, document_ids: [str], document_collection: str) -> List[int]:
        self.__cache_document_translation(document_collection)
        source2art = self.__doc_translation_source2art[document_collection]
        return [int(source2art[d]) for d in document_ids]
