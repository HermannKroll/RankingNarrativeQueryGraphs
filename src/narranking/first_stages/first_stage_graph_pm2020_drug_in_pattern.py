import itertools

from narraint.queryengine.engine import QueryEngine
from narraint.queryengine.query import GraphQuery, FactPattern
from narraplay.documentranking.benchmark import Benchmark, DRUG
from narraplay.documentranking.first_stages.first_stage_graph import FirstStageGraphRetriever
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.retriever import DocumentRetriever


class FirstStageGraphRetrieverPM2020DrugInPattern(FirstStageGraphRetriever):
    def __init__(self, name="FirstStageGraphRetrieverPM2020DrugInPattern"):
        super().__init__(name=name)
        self.retriever = DocumentRetriever()

    def retrieve_documents(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):
        # get full matching documents
        document_ids_with_score, statistics, doc2query2pos2prov_dict = super().retrieve_documents(query, collection,
                                                                                                  benchmark)
        if len(document_ids_with_score) == 0:
            return document_ids_with_score, statistics, doc2query2pos2prov_dict

        # convert to dictionary
        document_ids_with_score = {str(doc_id): score for (doc_id, score) in document_ids_with_score}
        statistics = {}
        if len(query.component2concepts) > 1:
            # Idea: given query (C1, C2, C3) => C1, C2, C3 are sets of entities
            # Graph (Query C1 associated C2) => 0.5
            # Graph (Query C1 associated C3) => 0.5
            # Graph (Query C2 associated C3) => 0.5

            concepts = list(query.component2concepts_with_type.values())
            # get all pairwise possible combinations (see schema above)
            for combination in itertools.combinations(concepts, r=2):
                # at least subject or object must be a Drug
                d1 = DRUG in {c.entity_type for c in combination[0]}
                d2 = DRUG in {c.entity_type for c in combination[1]}
                if not d1 and not d2:
                    continue

                graph_query = GraphQuery()
                graph_query.add_fact_pattern(FactPattern(combination[0], 'associated', combination[1]))

                results = QueryEngine.process_query_with_expansion(graph_query,
                                                                   document_collection_filter={collection},
                                                                   load_document_metadata=False)

                # Put in list and sort
                for res in results:
                    # if the document is contained it might have match the full pattern and
                    # thus has a higher score. So keep the original score and only add new
                    # documents here

                    doc_id = self.translate_document_id(res.document_id, res.document_collection, benchmark)
                    # Skip all documents that are not in the benchmark baseline
                    if not doc_id:
                        continue

                    if doc_id not in document_ids_with_score:
                        document_ids_with_score[doc_id] = 0.5

                        if len(res.position2provenance_ids) == 0:
                            raise ValueError(f'Document {doc_id} has no provenance information')

                        # Store provenance information
                        doc2query2pos2prov_dict[doc_id] = dict()
                        # this works because the new documents only match a single statement
                        doc2query2pos2prov_dict[doc_id][0] = res.position2provenance_ids

        document_ids_with_score = sorted([(doc, score) for doc, score in document_ids_with_score.items()],
                                         key=lambda x: (x[1], x[0]), reverse=True)

        # Count how often a score was assigned
        score2count = {}
        for d, score in document_ids_with_score:
            score_round = int(round(score * 100, 0))
            if score_round in score2count:
                score2count[score_round] += 1
            else:
                score2count[score_round] = 1

        statistics["all"] = {
            "document_ids": len(document_ids_with_score),
            "score2count": score2count
        }

        return document_ids_with_score, statistics, doc2query2pos2prov_dict
