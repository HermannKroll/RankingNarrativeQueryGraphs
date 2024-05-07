import itertools
import json
import os

from narraint.backend.database import SessionExtended
from narraint.backend.models import Predication
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RESULT_DIR_FIRST_STAGE
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.retriever import DocumentRetriever


class GraphFragment:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pos2prov_dict = self.read_pos2prov_from_file()

    def read_pos2prov_from_file(self):
        with open(self.file_path) as file:
            return json.load(file)

    def matches(self, query: AnalyzedQuery, document: AnalyzedNarrativeDocument):
        """
        Computes all distinct subgraph isomorphism between the query q and the document graph of d.
        Each subgraph isomorphism maps a part of the document graph to the query.
        Note that if q asks for two statements, each isomorphism must map two document edges to the
        corresponding query graph edges.

        Given two statements in a query:
        stmt1 maps to which edges of the document graph g? -> given by query engine through predication ids
        stmt2 maps to which edges of the document graph g? -> given by query engine through predication ids
        Cross product between all combinations
        """
        pos2prov_dict = self.pos2prov_dict[document.collection][str(query.topic.query_id)]
        if document.document_id_source not in pos2prov_dict:
            raise ValueError(f'Document {document.document_id_source} has no provenance information')

        # relevant document predications
        doc_predications = pos2prov_dict[document.document_id_source]

        document_predication_ids = set()
        for q_idx in doc_predications:
            for pos in doc_predications[q_idx]:
                document_predication_ids.update(doc_predications[q_idx][pos])
        # retrieve relevant predication mappings (id -> (subject, object))
        session = SessionExtended.get()
        q = session.query(Predication.id, Predication.subject_id, Predication.relation, Predication.object_id)
        q = q.filter(Predication.id.in_(document_predication_ids))
        predications = dict()
        for pid, s, p, o in q:
            predications[pid] = (s, p, o)

        # For each possible executed query
        fragments = list()
        for q_idx in doc_predications:
            # For each statement in that query
            doc_predications_per_position = list()
            for pos in doc_predications[q_idx]:
                # mappings for first query statement
                pos_doc_predications = list()
                for so in doc_predications[q_idx][pos]:
                    pos_doc_predications.append(predications[so])

                doc_predications_per_position.append(pos_doc_predications)

            # cross-product over all statements
            fragments.extend(list(itertools.product(*doc_predications_per_position)))

        # remove duplicated fragments
        fragments = list(set(fragments))
        return fragments


if __name__ == "__main__":
    concept_strategy = "hybrid"
    collection = "PubMed"
    retriever = DocumentRetriever()
    bench = Benchmark("trec-pm-2020-abstracts", "", [collection], load_from_file=True)

    path = os.path.join(RESULT_DIR_FIRST_STAGE, f'{bench.name}_FirstStageGraphRetriever_pos2prov.json')
    gf = GraphFragment(path)
    for idx, q in enumerate(bench.topics[:1]):  # FIXME use only the first topic for now
        print(str(q).center(100, "="))
        analyzed_query = AnalyzedQuery(q, concept_strategy=concept_strategy)
        doc_ids_to_rank = bench.topic_id2docs[q.query_id]

        print("Retrieve... ", end="")
        narrative_docs = list(retriever.retrieve_narrative_documents_for_collections(doc_ids_to_rank,
                                                                                     bench.document_collections))
        narrative_docs = sorted(narrative_docs, key=lambda x: x.document_id_source)
        print(f"{len(narrative_docs)} documents")

        for d in narrative_docs:
            d.prepare_with_min_confidence(0.0)
            fragment = gf.matches(analyzed_query, d)
            if fragment:
                print(fragment)
