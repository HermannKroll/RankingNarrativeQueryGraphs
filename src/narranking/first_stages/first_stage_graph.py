import itertools
import logging
from collections import defaultdict

from narraint.queryengine.engine import QueryEngine
from narraint.queryengine.query import GraphQuery, FactPattern
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.first_stages.first_stage_base import FirstStageBase
from narraplay.documentranking.query import AnalyzedQuery


class FirstStageGraphRetriever(FirstStageBase):
    def __init__(self, name="FirstStageGraphRetriever"):
        super().__init__(name=name)

    def retrieve_documents(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):
        statistics = {}
        document_ids_with_score = []
        score2count = {}
        doc2query2pos2prov_dict = dict()
        if len(query.component2concepts) > 1:
            # Idea: given query (C1, C2, C3) => C1, C2, C3 are sets of entities
            # Graph (Query C1 associated C2 AND C2 associated C3) => 1.0
            # Graph (Query C1 associated C2 AND C1 associated C3) => 1.0

            concepts = list(query.component2concepts_with_type.values())

            # store results
            results = []

            # 1. combine all concept pairs
            # 2. generate possible combinations between those sets (each set represents an edge)
            query_gen = itertools.combinations(itertools.combinations(concepts, r=2), r=len(concepts) - 1)
            for query_idx, concept_pair_edges in enumerate(query_gen):
                # generate a query for each pair combination set
                graph_query = GraphQuery()
                for concept_a, concept_b in concept_pair_edges:
                    graph_query.add_fact_pattern(FactPattern(concept_a, 'associated', concept_b))

                # execute the query now
                results_part = QueryEngine.process_query_with_expansion(graph_query,
                                                                        document_collection_filter={collection},
                                                                        load_document_metadata=False)
                # append results plus index
                results.append((query_idx, results_part))

            for query_idx, docs in results:
                for doc in docs:
                    doc_id = self.translate_document_id(doc.document_id, doc.document_collection, benchmark)
                    # Skip all documents that are not in the benchmark baseline
                    if not doc_id:
                        continue


                    doc2query2pos2prov_dict[doc_id] = defaultdict(dict)

                    for position, prov_ids in doc.position2provenance_ids.items():
                        if len(prov_ids) == 0:
                            logging.warning(f"Doc {doc.document_id} has no provenance for position {position}")
                        if position not in doc2query2pos2prov_dict[doc_id][query_idx]:
                            doc2query2pos2prov_dict[doc_id][query_idx][position] = list()
                        doc2query2pos2prov_dict[doc_id][query_idx][position].extend(prov_ids)

            # Extract document ids
            document_ids = doc2query2pos2prov_dict.keys()

            # Put in list and sort
            document_ids_with_score = sorted([(k, 1.0) for k in document_ids], key=lambda x: (x[1], x[0]), reverse=True)

            # Count how often a score was assigned
            score2count = {}
            for d, score in document_ids_with_score:
                score_round = int(round(score * 100, 0))
                if score_round in score2count:
                    score2count[score_round] += 1
                else:
                    score2count[score_round] = 1
        else:
            statistics["query"] = "no_concept_found"

        statistics["all"] = {
            "document_ids": len(document_ids_with_score),
            "score2count": score2count
        }

        return document_ids_with_score, statistics, doc2query2pos2prov_dict
