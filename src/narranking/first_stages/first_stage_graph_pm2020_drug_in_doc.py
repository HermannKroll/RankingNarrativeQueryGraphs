from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.first_stages.first_stage_partial_graph import FirstStagePartialGraphRetriever
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.retriever import DocumentRetriever


class FirstStageGraphRetrieverPM2020DrugInDoc(FirstStagePartialGraphRetriever):
    def __init__(self, name="FirstStageGraphRetrieverPM2020DrugInDoc"):
        super().__init__(name=name)
        self.retriever = DocumentRetriever()

    def retrieve_documents(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):
        # get full matching documents
        document_ids_with_score, statistics, doc2query2pos2prov_dict = super().retrieve_documents(query, collection,
                                                                                                  benchmark)
        if len(document_ids_with_score) == 0:
            return document_ids_with_score, statistics, doc2query2pos2prov_dict

        # get a list of document ids
        doc_ids = [d[0] for d in document_ids_with_score]
        docs = list(self.retriever.retrieve_narrative_documents_for_collections(doc_ids,
                                                                                benchmark.document_collections))

        relevant = set()
        # # apply filter
        drugs = {c for c in query.concepts if c.startswith('CHEMBL')}
        for doc in docs:
            if len(doc.concepts.intersection(drugs)) > 0:
                relevant.add(doc.document_id_source)

        print(f'Drug filter reduced documents from {len(doc_ids)} to {len(relevant)} documents')
        document_ids_with_score = sorted([(doc, score)
                                          for doc, score in document_ids_with_score
                                          if doc in relevant],
                                         key=lambda x: (x[1], x[0]), reverse=True)
        doc2query2pos2prov_dict = {k: v for k, v in doc2query2pos2prov_dict.items()
                                   if k in relevant}

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
