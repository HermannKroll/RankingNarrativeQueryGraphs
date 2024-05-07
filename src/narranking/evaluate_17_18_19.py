import json
import os
from collections import defaultdict
from typing import List

from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RESULT_DIR, EVAL_DIR, RESULT_DIR_FIRST_STAGE
from narraplay.documentranking.evaluate import load_results, load_document_count_statistics, calculate_table_data, \
    evaluate_runs
from narraplay.documentranking.first_stages.first_stage_partial_graph import FirstStagePartialGraphRetriever
from narraplay.documentranking.run_config import CONCEPT_STRATEGIES, SKIPPED_TOPICS, \
    EVALUATION_SKIP_BAD_TOPICS

REPORT_JUST_METHODS_IN_PAPER = True

BENCHMARKS = [
    Benchmark("trec-pm-2017-abstracts", "medline/2017/trec-pm-2017", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2018-abstracts", "medline/2017/trec-pm-2018", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2019-abstracts", "clinicaltrials/2019/trec-pm-2019", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-covid-rnd5-abstracts", "cord19/trec-covid/round5", ["trec_covid_round5_abstract"]),
    Benchmark("trec-covid-rnd5-fulltexts", "cord19/trec-covid/round5", ["trec_covid_round5_abstract_fulltext"],
              has_fulltexts=True),
]

BENCHMARK_2_REAL_NAME = {
    "trec-pm-2017-abstracts": "TREC-PM 2017",
    "trec-pm-2018-abstracts": "TREC-PM 2018",
    "trec-pm-2019-abstracts": "TREC-PM 2019",
    "trec-covid-rnd5-abstracts": "TREC COVID 2020",
    "trec-covid-rnd5-fulltexts": "TREC COVID 2020",
}

METHODS_TO_REPORT_IN_PAPER_PM = {
    "EqualDocumentRanker": "Partial Match",
    "WeightedDocumentRanker_min-0": "GraphRank",
    "BM25 Native": "Native BM25 (Baseline)"
}

METHODS_TO_REPORT_IN_PAPER_TC = {
    "EqualDocumentRanker": "",
    "WeightedDocumentRanker_min-0": "GraphRank",
    "BM25Text": "BM25",
    "BM25 Native": "Native BM25"
}

RESULT_MEASURES = {
    'recall_1000': 'Recall@1000',

    'ndcg_cut_10': 'nDCG@10',
    'ndcg_cut_20': 'nDCG@20',

    'P_10': 'P@10',
    'P_20': 'P@20',
}


def generate_table(measures: dict, strategies: str, first_stage: str, benchmarks: List[Benchmark],
                   min_docs_per_topic: int = None, min_recall: float = 0.0):
    score_rows_bms = defaultdict(defaultdict)
    measures = [(k, v) for k, v in measures.items()]

    empty_measures = {k: "-" for k, _ in measures}  # empty entries for table generation
    relevant_rankers_pm = set(METHODS_TO_REPORT_IN_PAPER_PM.keys())
    relevant_rankers_tc = set(METHODS_TO_REPORT_IN_PAPER_TC.keys())

    for benchmark in benchmarks:
        for strategy in strategies:
            if strategy == "likesimilarityontology" and "trec-covid" in benchmark.name:
                continue

            print("--" * 60)
            print(f"Generating score table for {first_stage} with benchmarks {benchmark.name}")

            path = os.path.join(EVAL_DIR, f"{first_stage}_{strategy}_", f"{benchmark.name}")
            results = load_results(path)

            skipped_topics = SKIPPED_TOPICS[benchmark.name].copy()

            if EVALUATION_SKIP_BAD_TOPICS:
                # Load statistics concerning translation and components
                stats_dir = os.path.join(RESULT_DIR_FIRST_STAGE, 'statistics')
                stats_path = os.path.join(stats_dir, f'{benchmark.name}_{first_stage}_{strategy}.json')
                with open(stats_path, 'rt') as f:
                    stats = json.load(f)

                for topic_id in stats:
                    if 'skipped' in stats[topic_id]:
                        skipped_topics.append(topic_id)

            relevant_topics = {str(q.query_id) for q in benchmark.topics
                               if str(q.query_id) not in skipped_topics}
            print('==' * 60)
            print(f'Evaluation based on {len(relevant_topics)} topics')
            print('==' * 60)
            results = [(k, v) for k, v in sorted(results.items(), key=lambda x: x[0])]

            if min_docs_per_topic:
                path = os.path.join(RESULT_DIR, "statistics", f"{benchmark.name}_{first_stage}_{strategy}.json")
                doc_count = load_document_count_statistics(path)
                topics_with_less_docs = {q_id for q_id, doc_c in doc_count if doc_c < min_docs_per_topic}
                relevant_topics = relevant_topics.difference(topics_with_less_docs)
                print(f"Topics with less than {min_docs_per_topic} docs: {topics_with_less_docs}")
                print(f"{len(relevant_topics)} relevant topics: {relevant_topics}")

            if min_recall:
                topics_with_less_recall = set()
                for r_name, r_result in results:
                    for q, scores in r_result.items():
                        if scores['set_recall'] < min_recall:
                            topics_with_less_recall.add(q)

                relevant_topics = relevant_topics.difference(topics_with_less_recall)
                print(f"Topics with less than {min_recall} recall docs: {topics_with_less_recall}")
                print(f"{len(relevant_topics)} relevant topics: {relevant_topics}")

            statistics = {
                "benchmark_topics": len(benchmark.topics),
                "relevant_topics": len(relevant_topics),
                "name": BENCHMARK_2_REAL_NAME[benchmark.name]
            }

            if benchmark.name.startswith("trec-pm-"):
                score_rows, _ = calculate_table_data(measures, results, relevant_topics, relevant_rankers_pm)
            else:
                score_rows, _ = calculate_table_data(measures, results, relevant_topics, relevant_rankers_tc)
            if strategy not in score_rows_bms[benchmark.name]:
                score_rows_bms[benchmark.name][strategy] = (score_rows, statistics)

    print('==' * 60)
    print("--" * 60)
    print("Creating table content")
    print("--" * 60)

    # create tabular LaTeX code
    rows = list()
    rows.append("%%%%% begin autogenerated %%%%%")
    rows.append(r"\begin{tabular}{lccccc}")
    rows.append(r"\toprule")
    rows.append("Strategy & " + " & ".join([*(str(m[1]) for m in measures)]) + r"\\")

    for benchmark in benchmarks:
        benchmark_stats = score_rows_bms[benchmark.name][strategies[0]][1]
        title = (rf"{benchmark_stats['name']} - Topics ({benchmark_stats['relevant_topics']}/"
                 rf"{benchmark_stats['benchmark_topics']}) - ")
        if "fulltexts" in benchmark.name:
            title += "Fulltexts"
        else:
            title += "Abstracts"
        rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{6}}{{c}}{{\textbf{{{title}}}}} \\")
        rows.append(r"\midrule")

        max_m = {k: 0.0 for k in RESULT_MEASURES}
        # merge max values over all strategies
        for strategy in strategies:
            if strategy not in score_rows_bms[benchmark.name]:
                continue

            score_rows, _ = score_rows_bms[benchmark.name][strategy]
            for m in max_m.keys():
                max_m[m] = max([max_m[m], *(sv[m] for k, sv in score_rows.items())])

        if "trec-pm" in benchmark.name:
            # like
            score_rows, _ = score_rows_bms[benchmark.name]["likesimilarity"]
            row = "Partial Match & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["EqualDocumentRanker"].items())
            row += r" \\"
            rows.append(row)

            row = r"\quad + GraphRank & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["WeightedDocumentRanker_min-0"].items())
            row += r" \\"
            rows.append(row)

            # like + ontology
            score_rows, _ = score_rows_bms[benchmark.name]["likesimilarityontology"]
            row = r"\quad + Ontology + GraphRank & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["WeightedDocumentRanker_min-0"].items())
            row += r" \\"
            rows.append(row)

            row = r"Native BM25 (Baseline) & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["BM25 Native"].items())
            row += r" \\"
            rows.append(row)
        else:
            score_rows, _ = score_rows_bms[benchmark.name]["likesimilarity"]
            row = "Partial Match & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["EqualDocumentRanker"].items())
            row += r" \\"
            rows.append(row)

            row = r"\quad + GraphRank & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["WeightedDocumentRanker_min-0"].items())
            row += r" \\"
            rows.append(row)

            row = r"\quad + BM25 & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["BM25Text"].items())
            row += r" \\"
            rows.append(row)

            row = r"Native BM25 (Baseline) & "
            row += " & ".join(rf"\textbf{{{str(s)}}}" if max_m[m] == s else str(s)
                              for m, s in score_rows["BM25 Native"].items())
            row += r" \\"
            rows.append(row)

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append("%%%%% end autogenerated %%%%%")

    print("\n".join(rows))
    print("--" * 60)


if __name__ == "__main__":
    evaluate_runs(with_bm25_native=True, benchmarks=BENCHMARKS)
    generate_table(RESULT_MEASURES, CONCEPT_STRATEGIES, FirstStagePartialGraphRetriever().name, BENCHMARKS)
