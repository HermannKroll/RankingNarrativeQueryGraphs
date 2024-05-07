from collections import defaultdict
from typing import List, Set

from sqlalchemy import and_

from kgextractiontoolbox.backend.models import Document, Tag, Predication
from kgextractiontoolbox.backend.retrieve import iterate_over_all_documents_in_collection
from kgextractiontoolbox.document.document import TaggedEntity
from narraint.backend.database import SessionExtended
from kgextractiontoolbox.document.narrative_document import NarrativeDocument, StatementExtraction
from narrant.entity.entityresolver import GeneResolver
from narrant.entitylinking.enttypes import GENE
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.translator import DocumentTranslator


def retrieve_narrative_documents_from_database_small(session, document_ids: Set[int], document_collection: str) \
        -> List[NarrativeDocument]:
    """
    Retrieves a set of Narrative Documents from the database
    :param session: the current session
    :param document_ids: a set of document ids
    :param document_collection: the corresponding document collection
    :return: a list of NarrativeDocuments
    """
    doc_results = {}

    # logging.info(f'Querying {len(document_ids)} from collection: {document_collection}...')
    # first query document titles and abstract
    doc_query = session.query(Document).filter(and_(Document.id.in_(document_ids),
                                                    Document.collection == document_collection))

    for res in doc_query:
        doc_results[res.id] = NarrativeDocument(document_id=res.id, title=res.title, abstract=res.abstract)

    if len(doc_results) != len(document_ids):
        diff = document_ids - doc_results.keys()
        raise ValueError(f'Did not retrieve all required {document_collection} documents (missed ids: {diff})')

    #  logging.info('Querying for tags...')
    # Next query for all tagged entities in that document
    tag_query = session.query(Tag).filter(and_(Tag.document_id.in_(document_ids),
                                               Tag.document_collection == document_collection))
    tag_result = defaultdict(list)
    for res in tag_query:
        tag_result[res.document_id].append(TaggedEntity(document=res.document_id,
                                                        start=res.start,
                                                        end=res.end,
                                                        ent_id=res.ent_id,
                                                        ent_type=res.ent_type,
                                                        text=res.ent_str))
    for doc_id, tags in tag_result.items():
        doc_results[doc_id].tags = tags
        doc_results[doc_id].sort_tags()

    # logging.info('Querying for statement extractions...')
    # Next query for extracted statements
    es_query = session.query(Predication)
    es_query = es_query.filter(Predication.document_collection == document_collection)
    es_query = es_query.filter(Predication.document_id.in_(document_ids))
    es_query = es_query.filter(Predication.relation != None)

    es_for_doc = defaultdict(list)
    sentence_ids = set()
    sentenceid2doc = defaultdict(set)
    for res in es_query:
        es_for_doc[res.document_id].append(StatementExtraction(subject_id=res.subject_id,
                                                               subject_type=res.subject_type,
                                                               subject_str=res.subject_str,
                                                               predicate=res.predicate,
                                                               relation=res.relation,
                                                               object_id=res.object_id,
                                                               object_type=res.object_type,
                                                               object_str=res.object_str,
                                                               sentence_id=res.sentence_id,
                                                               confidence=res.confidence))
        sentence_ids.add(res.sentence_id)
        sentenceid2doc[res.sentence_id].add(res.document_id)

    for doc_id, extractions in es_for_doc.items():
        doc_results[doc_id].extracted_statements = extractions

    return list(doc_results.values())


class DocumentRetriever:

    def __init__(self):
        self.__cache = {}
        self.translator = DocumentTranslator()
        self.session = SessionExtended.get()
        self.generesolver = GeneResolver()
        self.generesolver.load_index()

    def retrieve_document_ids_for_collection(self, document_collection: str):
        session = SessionExtended.get()
        q = session.query(Document.id).filter(Document.collection == document_collection)
        doc_ids = set()
        for d in q:
            doc_ids.add(d[0])
        return doc_ids

    def retrieve_documents_text(self, document_ids: [int], document_collection: str):
        session = SessionExtended.get()
        doc_texts = []
        for doc in iterate_over_all_documents_in_collection(session, document_ids=document_ids,
                                                            collection=document_collection,
                                                            consider_sections=True):
            doc_texts.append((doc.id, doc.get_text_content(sections=True)))
        return doc_texts

    def retrieve_narrative_documents_for_collections(self, document_ids: [str], document_collections: [str]):
        # This trick does work because the collections have different document ids,
        # i.e. each id belongs to a unique collection
        if len(document_collections) == 1:
            yield from self.retrieve_narrative_documents(document_ids, document_collections[0])
        elif len(document_collections) == 2:
            # multiple collections are queried. divide ids for each collection
            # hack: pubmed ids are integers. other ids are not integers
            assert "PubMed" in document_collections
            pubmed_ids = []
            other_ids = []
            for did in document_ids:
                try:
                    pubmed_ids.append(int(did))
                except ValueError:
                    other_ids.append(did)

            # get the name of the other collection
            c = [dc for dc in document_collections if dc != "PubMed"][0]
            yield from self.retrieve_narrative_documents(pubmed_ids, "PubMed")
            yield from self.retrieve_narrative_documents(other_ids, c)
        else:
            raise ValueError(f'Do not support retrieval from {len(document_collections)} collections')

    def retrieve_narrative_documents(self, document_ids: [str], document_collection: str, translate_ids=True) -> List[
        AnalyzedNarrativeDocument]:
        if len(document_ids) == 0:
            return []

        if translate_ids:
            # First translate the document ids
            # Hack: PubMed does not need to be translated
            if document_collection == 'PubMed':
                translated = []
                # Translate all integer pubmed ids
                for did in document_ids:
                    try:
                        translated.append(int(did))
                    except ValueError:
                        pass
                document_ids = translated
            else:
                print(f'Should translate {len(document_ids)} ids...')
                document_ids = self.translator.translate_document_ids_source2art(document_ids, document_collection)
                print(f'{len(document_ids)} document ids translated...')
        else:
            # just make them integers
            document_ids = {int(d) for d in document_ids}

        document_ids = set(document_ids)
        found_ids = set()
        narrative_documents = []

        if document_collection not in self.__cache:
            self.__cache[document_collection] = {}

        # look which documents have been cached
        for did in document_ids:
            if did in self.__cache[document_collection]:
                found_ids.add(did)
                narrative_documents.append(self.__cache[document_collection][did])

        remaining_document_ids = document_ids - found_ids
        if len(remaining_document_ids) == 0:
            assert len(narrative_documents) == len(document_ids)
            return narrative_documents
        narrative_documents_queried = retrieve_narrative_documents_from_database_small(session=self.session,
                                                                                       document_ids=remaining_document_ids,
                                                                                       document_collection=document_collection)
        # Gene IDs are only present in the Tag table.
        # The rest work with gene symbols
        for doc in narrative_documents_queried:
            self.__translate_gene_ids_to_symbols(doc)

        narrative_documents_queried = [
            AnalyzedNarrativeDocument(d, d.id,
                                      self.translator.translate_document_id_art2source(d.id, document_collection),
                                      collection=document_collection)
            for d in narrative_documents_queried]

        # add to cache
        for d in narrative_documents_queried:
            self.__cache[document_collection][d.document.id] = d

        # add them to list
        narrative_documents.extend(narrative_documents_queried)
        assert len(narrative_documents) == len(document_ids)
        return narrative_documents

    def __translate_gene_ids_to_symbols(self, document: NarrativeDocument):
        translated_gene_ids = []
        for tag in document.tags:
            # Gene IDs need a special handling
            if tag.ent_type == GENE:
                if ';' in tag.ent_id:
                    for g_id in tag.ent_id.split(';'):
                        try:
                            symbol = self.generesolver.gene_id_to_symbol(g_id.strip()).lower()
                            translated_gene_ids.append(TaggedEntity(document=tag.document,
                                                                    start=tag.start,
                                                                    end=tag.end,
                                                                    text=tag.text,
                                                                    ent_id=symbol,
                                                                    ent_type=GENE))
                        except (KeyError, ValueError):
                            continue
                else:
                    try:
                        symbol = self.generesolver.gene_id_to_symbol(tag.ent_id).lower()
                        translated_gene_ids.append(TaggedEntity(document=tag.document,
                                                                start=tag.start,
                                                                end=tag.end,
                                                                text=tag.text,
                                                                ent_id=symbol,
                                                                ent_type=GENE))
                    except (KeyError, ValueError):
                        pass

        # remove all gene types
        document.tags = [t for t in document.tags if t.ent_type != GENE]
        # add translated genes
        document.tags.extend(translated_gene_ids)
