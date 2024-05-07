from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.translator import DocumentTranslator


class FirstStageBase:
    def __init__(self, name):
        self.name = name
        self.translator = DocumentTranslator()

    def retrieve_documents(self, query: AnalyzedQuery, collection: [str], benchmark: Benchmark):
        raise NotImplementedError()

    def translate_document_id(self, document_id: int, document_collection: str, benchmark: Benchmark):
        # Skip all documents that are not in the benchmark baseline
        doc_id = self.translator.translate_document_id_art2source(document_id, document_collection)
        if document_collection == "PubMed" and benchmark.get_documents_for_baseline():
            if int(doc_id) not in benchmark.get_documents_for_baseline():
                return None
        return str(doc_id)
