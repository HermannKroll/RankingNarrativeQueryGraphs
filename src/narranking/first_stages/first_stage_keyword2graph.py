import logging
from collections import defaultdict

from narraint.keywords2graph.translation import Keyword2GraphTranslation, SupportedGraphPattern
from narraint.queryengine.engine import QueryEngine
from narraint.queryengine.query import GraphQuery, FactPattern
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.first_stages.first_stage_base import FirstStageBase
from narraplay.documentranking.query import AnalyzedQuery


class FirstStageKeyword2GraphRetriever(FirstStageBase):
    def __init__(self, name="FirstStageKeyword2GraphRetriever"):
        super().__init__(name=name)
        self.keyword2graph = Keyword2GraphTranslation()

    def retrieve_documents(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):

        keywords = query.component2concepts.keys()

        statistics = {}
        doc2query2pos2prov_dict = dict()
        document_ids_with_score = dict()
        try:
            patterns = self.keyword2graph.translate_keywords(keyword_lists=list(keywords))

            if len(patterns) > 0:
                score_diff = 1.0 / len(patterns)
            else:
                score_diff = 0.0
            for query_idx, pattern in enumerate(patterns):
                # the actual score for this pattern is based on its position
                # 3 patterns: the first pattern receives a score of 1.0, the second 0.66 and the third of 0.33
                # 4 patterns: 1. -> 1.0, 2. -> 0.75, 3. -> 0.50 and 4. -> 0.25
                pattern_score = 1.0 - (score_diff * query_idx)

                graph_query = GraphQuery()
                for supported_fp in pattern.fact_patterns:
                    fp = FactPattern(subjects=query.component2concepts_with_type[supported_fp.keyword1],
                                     predicate=supported_fp.relation,
                                     objects=query.component2concepts_with_type[supported_fp.keyword2])
                    graph_query.add_fact_pattern(fp)

                # Get results 1
                results = QueryEngine.process_query_with_expansion(graph_query, document_collection_filter={collection},
                                                                   load_document_metadata=False)

                for doc in results:
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

                for doc_id in document_ids:
                    # does the document already have a higher score?
                    if doc_id in document_ids_with_score:
                        continue
                    # no -> then it receives the actual pattern score
                    document_ids_with_score[doc_id] = pattern_score

            # Convert the scores to a list and sort them desc
            document_ids_with_score = sorted([(k, v) for k, v in document_ids_with_score.items()],
                                             key=lambda x: (x[1], x[0]),
                                             reverse=True)

            # Count how often a score was assigned
            score2count = {}
            for d, score in document_ids_with_score:
                score_round = int(round(score * 100, 0))
                if score_round in score2count:
                    score2count[score_round] += 1
                else:
                    score2count[score_round] = 1

            return document_ids_with_score, statistics, doc2query2pos2prov_dict

        except ValueError:
            return [], dict(), dict()
