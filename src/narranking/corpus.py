import logging
import math

from tqdm import tqdm

from kgextractiontoolbox.backend.models import Document
from narraint.backend.database import SessionExtended
from narraint.backend.models import PredicationInvertedIndex, TagInvertedIndex
from narrant.cleaning.pharmaceutical_vocabulary import SYMMETRIC_PREDICATES


class DocumentCorpus:

    def __init__(self, collections: [str]):
        self.collections = collections

        logging.info(f'Estimating size of document corpus (collections = {self.collections})')
        session = SessionExtended.get()
        self.document_count = 0
        for collection in collections:
            logging.info(f'Counting documents in collection: {collection}')
            col_count = session.query(Document.id).filter(Document.collection == collection).count()
            self.document_count += col_count
            logging.info(f'{col_count} documents found')

        logging.info(f'{self.document_count} documents in corpus')
        self.cache_statement2count = dict()
        self.cache_concept2support = dict()
        self.all_idf_data_cached = False
        self.load_all_support_into_memory()


    def load_all_support_into_memory(self):
        session = SessionExtended.get()
        # print('Caching all predication inverted index support entries...')
        # total = session.query(PredicationInvertedIndex).count()
        # q = session.query(PredicationInvertedIndex.subject_id,
        #                   PredicationInvertedIndex.relation,
        #                   PredicationInvertedIndex.object_id,
        #                   PredicationInvertedIndex.support)
        # for row in tqdm(q, desc="Loading db data...", total=total):
        #     statement = row.subject_id, row.relation, row.object_id
        #     self.cache_statement2count[statement] = row.support
        print('Caching all concept inverted index support entries...')
        total = session.query(TagInvertedIndex).count()
        q = session.query(TagInvertedIndex.entity_id,
                          TagInvertedIndex.document_collection,
                          TagInvertedIndex.support)
        for row in tqdm(q, desc="Loading db data...", total=total):
            if row.document_collection in self.collections:
                if row.entity_id in self.cache_concept2support:
                    self.cache_concept2support[row.entity_id] += row.support
                else:
                    self.cache_concept2support[row.entity_id] = row.support
        self.all_idf_data_cached = True
        print('Finished')

    def get_idf_score(self, statement: tuple):
        return math.log(self.get_document_count() / self.get_statement_documents(statement))

    def get_concept_ifd_score(self, entity_id: str):
        return math.log(self.get_document_count() / self.get_concept_support(entity_id))

    def get_document_count(self):
        return self.document_count

    def _get_statement_documents_without_symmetric(self, statement: tuple):
        # number of documents which support the statement
        if statement in self.cache_statement2count:
            return self.cache_statement2count[statement]

        # not in index, but all data should be loaded. so no retrieval is needed any more
        # however, some strange statement concept might not appear in the concept index
        if self.all_idf_data_cached:
            return 1

        session = SessionExtended.get()
        q = session.query(PredicationInvertedIndex.support)
        if len(self.collections) == 1:
            q = q.filter(PredicationInvertedIndex.document_collection == next(iter(self.collections)))
        else:
            q = q.filter(PredicationInvertedIndex.document_collection.in_(self.collections))
        q = q.filter(PredicationInvertedIndex.subject_id == statement[0])
        q = q.filter(PredicationInvertedIndex.relation == statement[1])
        q = q.filter(PredicationInvertedIndex.object_id == statement[2])

        support = 0
        for row in q:
            support += row.support

        self.cache_statement2count[statement] = support
        return support

    def get_statement_documents(self, statement: tuple):
        if statement[1] in SYMMETRIC_PREDICATES:
            support = (self._get_statement_documents_without_symmetric(statement) +
                       self._get_statement_documents_without_symmetric((statement[2], statement[1], statement[0])))
        else:
            support = self._get_statement_documents_without_symmetric(statement)

        assert support > 0
        return support

    def get_concept_support(self, entity_id):
        if entity_id in self.cache_concept2support:
            return self.cache_concept2support[entity_id]

        # not in index, but all data should be loaded. so no retrieval is needed any more
        # however, some strange statement concept might not appear in the concept index
        if self.all_idf_data_cached:
            return 1

        session = SessionExtended.get()
        q = session.query(TagInvertedIndex.support)
        q = q.filter(TagInvertedIndex.entity_id == entity_id)
        support = 0
        for row in q:
            support += row.support

        self.cache_concept2support[entity_id] = support
        # assert support > 0

        return support
