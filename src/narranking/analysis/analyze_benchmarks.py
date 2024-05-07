import json
import os
from datetime import datetime

from tqdm import tqdm

from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.retriever import DocumentRetriever
from narraplay.documentranking.run_config import BENCHMARKS, CONCEPT_STRATEGIES


def num_topics(benchmark: Benchmark):
    return len(benchmark.topics)


def num_topics_with_2_or_more_concepts(benchmark: Benchmark, concept_strategy):
    relevant_topics = 0
    for q in benchmark.topics:
        analyzed_query = AnalyzedQuery(q, concept_strategy=concept_strategy)
        if len(analyzed_query.concepts) >= 2:
            relevant_topics += 1
    return relevant_topics


def num_relevant_docs(benchmark: Benchmark):
    return len(benchmark.get_relevant_documents())


def num_relevant_docs_with_graph(retriever: DocumentRetriever, benchmark: Benchmark):
    relevant_docs = benchmark.get_relevant_documents()
    document_generator = retriever.retrieve_narrative_documents_for_collections(relevant_docs,
                                                                                benchmark.document_collections)
    docs_with_graph = 0
    docs_retrieved = 0
    progress = tqdm(len(relevant_docs), desc='Processing')
    for doc in document_generator:
        doc.prepare_with_min_confidence(0.0)
        if len(doc.graph) > 0:
            docs_with_graph += 1
        docs_retrieved += 1
        progress.update(docs_retrieved)
    return docs_with_graph, docs_retrieved


def main():
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    retriever = DocumentRetriever()
    for benchmark in BENCHMARKS:
        data = dict()
        print("==" * 60)
        print("==" * 60)
        print("Benchmark", benchmark.name)
        start = datetime.now()

        data["no_topics"] = num_topics(benchmark)
        data["no_topics_with_2_or_more_concepts"] = (
            num_topics_with_2_or_more_concepts(benchmark, CONCEPT_STRATEGIES[0]))
        data["relevant_docs"] = num_relevant_docs(benchmark)
        docs_with_graph, docs_retrieved = num_relevant_docs_with_graph(retriever, benchmark)
        data["relevant_docs_retrieved"] = docs_retrieved
        data["relevant_docs_with_graph"] = docs_with_graph

        with open(os.path.join("tmp", f"{benchmark.name}_stats.json"), "wt") as outfile:
            json.dump(data, outfile)
        print("finished after", datetime.now() - start)


if __name__ == "__main__":
    main()
