from narraint.keywords2graph.translation import SupportedGraphPattern
from narraint.queryengine.engine import QueryEngine
from narraint.queryengine.query import GraphQuery, FactPattern
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.first_stages.first_stage_keyword2graph import FirstStageKeyword2GraphRetriever
from narraplay.documentranking.query import AnalyzedQuery


class FirstStageKeyword2PartialGraphRetriever(FirstStageKeyword2GraphRetriever):
    def __init__(self, name="FirstStageKeyword2PartialGraphRetriever"):
        super().__init__(name=name)

    def retrieve_documents(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):
        keywords = query.component2concepts.keys()
        try:
            patterns: [SupportedGraphPattern] = self.keyword2graph.translate_keywords(keyword_lists=list(keywords))

            # get full matching documents
            document_ids_with_score, statistics, doc2query2pos2prov_dict = super().retrieve_documents(query, collection,
                                                                                                      benchmark)
            # we need to rescale the pattern scores of the first stage
            # e.g. documents might have a score of 1.0, 0.66 and 0.33
            # we rescale them to 1.0, 0.80 and 0.6

            # convert to dictionary and scale the scores
            document_ids_with_score = {str(doc_id): ((score * 0.5) + 0.5) for (doc_id, score) in
                                       document_ids_with_score}

            # scale these patterns between 0.5 and 0
            if len(patterns) > 0:
                score_diff = 0.5 / len(patterns)
            else:
                score_diff = 0.0
            for idx, pattern in enumerate(patterns):
                # the actual score for this pattern is based on its position
                # 3 patterns: the first pattern receives a score of 1.0, the second 0.66 and the third of 0.33
                # 4 patterns: 1. -> 1.0, 2. -> 0.75, 3. -> 0.50 and 4. -> 0.25
                pattern_score = 0.5 - (score_diff * idx)
                # Search for all proposed parts one by one
                for supported_fp in pattern.fact_patterns:
                    fp = FactPattern(subjects=query.component2concepts_with_type[supported_fp.keyword1],
                                     predicate=supported_fp.relation,
                                     objects=query.component2concepts_with_type[supported_fp.keyword2])

                    graph_query = GraphQuery()
                    graph_query.add_fact_pattern(fp)

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
                            document_ids_with_score[doc_id] = pattern_score

                            if len(res.position2provenance_ids) == 0:
                                raise ValueError(f'Document {doc_id} has no provenance information')

                            # Store provenance information
                            doc2query2pos2prov_dict[doc_id] = dict()
                            # this works because the new documents only match a single statement
                            doc2query2pos2prov_dict[doc_id][0] = res.position2provenance_ids

            # Count how often a score was assigned
            score2count = {}
            for d, score in document_ids_with_score.items():
                score_round = int(round(score * 100, 0))
                if score_round in score2count:
                    score2count[score_round] += 1
                else:
                    score2count[score_round] = 1

            statistics["all"] = {
                "document_ids": len(document_ids_with_score),
                "score2count": score2count
            }

            document_ids_with_score = sorted([(k, v) for k, v in document_ids_with_score.items()],
                                             key=lambda x: (x[1], x[0]), reverse=True)
            return document_ids_with_score, statistics, doc2query2pos2prov_dict

        except ValueError:
            return [], dict(), dict()
