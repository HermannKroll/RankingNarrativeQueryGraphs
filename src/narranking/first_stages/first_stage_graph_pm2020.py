import ast

from narrant.entitylinking.enttypes import SPECIES
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import PM2020_MAMMELS_TAX_ID_FILE
from narraplay.documentranking.first_stages.first_stage_graph import FirstStageGraphRetriever
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.retriever import DocumentRetriever


class FirstStageGraphRetrieverPM2020(FirstStageGraphRetriever):
    def __init__(self, name="FirstStageGraphRetrieverPM2020"):
        super().__init__(name=name)
        self.retriever = DocumentRetriever()
        self.concept_filter = set()
        with open(PM2020_MAMMELS_TAX_ID_FILE, 'rt') as f:
            self.concept_filter = ast.literal_eval(f.read())
        self.concept_filter = {int(s) for s in self.concept_filter}
        print(f'{len(self.concept_filter)} mammel ids load')

    def retrieve_documents(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):
        # get full matching documents
        document_ids_with_score, statistics, doc2query2pos2prov_dict = super().retrieve_documents(query, collection,
                                                                                                  benchmark)
        if len(document_ids_with_score) == 0:
            return document_ids_with_score, statistics, doc2query2pos2prov_dict


        doc_ids = [d[0] for d in document_ids_with_score]
        docs = list(self.retriever.retrieve_narrative_documents_for_collections(doc_ids,
                                                                                benchmark.document_collections))

        relevant = set()
        # apply filter
        for doc in docs:
            doc_species = {int(tag.ent_id) for tag in doc.document.tags if tag.ent_type == SPECIES}
            # check whether the document is about a human or about a mammel
            if len(doc_species.intersection(self.concept_filter)) > 0:
                relevant.add(doc.document_id_source)

        document_ids_with_score = [d for d in document_ids_with_score if d[0] in relevant]
        doc2query2pos2prov_dict = {k: v for k, v in doc2query2pos2prov_dict.items() if k in relevant}

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

        print(f'Human/Mammel filter reduced documents from {len(doc_ids)} to {len(relevant)} documents')

        return document_ids_with_score, statistics, doc2query2pos2prov_dict
