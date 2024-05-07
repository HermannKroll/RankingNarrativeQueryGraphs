import glob
import json
import os
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytrec_eval as pe

from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RESULT_DIR, RUNS_DIR, EVAL_DIR, RESULT_DIR_FIRST_STAGE, QRELS_PATH, \
    RESULT_DIR_BASELINES
from narraplay.documentranking.run_config import FIRST_STAGE_NAMES, CONCEPT_STRATEGIES, BENCHMARKS, SKIPPED_TOPICS, \
    EVALUATION_SKIP_BAD_TOPICS, JUDGED_DOCS_ONLY_FLAG

REPORT_JUST_METHODS_IN_PAPER = True

METHODS_TO_REPORT_IN_PAPER = {
    "Equal": "No Ranking",
    "ConceptCoverage": "Coverage",
    "Confidence": "Confidence",
    "RelationalSim": "Relational Similarity",
    #   "TfIdfAvg": "tf-IDF-Avg",
    #   "TfIdfMax": "tf-IDF-Max",
    "TfIdfMin": "tf-IDF-Min",
    # "Weighted avg": "Weighted Strategy avg",
    # "Weighted max": "Weighted Strategy max",
    "Weighted min": "Weighted Strategy min",
    "BM25Text": "BM25 Reranking",
    "BM25 Native": "BM25 Native"
}

MEASURES = [
    'map',
    'set_recall',
    'num_ret',
    'recall_10',
    'recall_100',
    'recall_500',
    'recall_1000',
    'recall_2500',
    'recall_5000',
    'ndcg_cut_10',
    'ndcg_cut_20',
    'ndcg_cut_30',
    'ndcg_cut_100',
    'bpref',
    'ndcg',
    'P_100',
    'P_20',
    'P_10',
    "Rprec"
    #  "rbp"
]

FS_RESULT_MEASURES = {
    'num_ret': 'Retrieved',
    'recall_100': 'Recall@100',
    'recall_500': 'Recall@500',
    'recall_1000': 'Recall@1000',
    #  'recall_2500': 'Recall@2500',
    #  'recall_5000': 'Recall@5000',
    'set_recall': 'Recall@All'
}

RESULT_MEASURES = {
    'recall_1000': 'Recall@1000',
    #  'set_recall': 'Recall@All',
    'ndcg_cut_10': 'nDCG@10',
    'ndcg_cut_20': 'nDCG@20',
    'ndcg_cut_100': 'nDCG@100',
    #    'map': 'MAP',
    'P_10': 'P@10',
    'P_20': 'P@20',
    "P_100": 'P@100',
    # "Rprec": "Rprec",
    # "ndcg": "ndcg",
    # 'ndcg_cut_30': 'nDCG@30',

    # "rbp": "rbp"
    # "bpref": "bpref"
}


def load_qrel_from_file(path: str):
    with open(path, 'r') as file:
        return pe.parse_qrel(file)


def load_run_from_file(path: str):
    with open(path, 'r') as file:
        return pe.parse_run(file)


def extract_run(run: dict, measure: str):
    run = sorted(run.items(), key=lambda x: int(x[0]))
    indices = [k for k, _ in run]
    values = [(v[measure],) for _, v in run]

    return {i: v[0] for i, v in zip(indices, values)}


def generate_diagram(measures: dict, strategy: str, first_stage: str, benchmark: str, results: dict = None):
    path = os.path.join(EVAL_DIR, f"{first_stage}_{strategy}_", f"{benchmark}")
    if not results:
        results = load_results(path)

    for measure in measures:
        dfs = [pd.DataFrame.from_dict(extract_run(run, measure), orient="index", columns=[name])
               for name, run in results.items()]
        df = pd.concat(dfs, axis=1)
        measure = measure.replace("_cut", "").replace("_", "@")
        fig = plt.figure()
        _ = df.plot.bar(xlabel=f'Topic', ylabel=measure, figsize=(40, 10), width=1.0, edgecolor='black')
        plt.savefig(os.path.join(path, f"{measure}.svg"), format="svg")
        plt.close(fig)


def calculate_table_data(measures: List[tuple], results: List[tuple], relevant_topics: set,
                         relevant_rankers: set = None):
    """
    Calculate the mean scores of the given measures for each ranker. The results may contain ranker results, that are
    not relevant for this evaluation. They are ignored in the further processing of the table data and calculation of
    the max values for each measure.
    """
    max_m = {m[0]: 0.0 for m in measures}
    score_rows = defaultdict(dict)
    for name, raw_run in results:
        if relevant_rankers and name not in relevant_rankers:
            continue
        s_row = dict()
        no_retrieved = extract_run(raw_run, "num_ret")
        no_retrieved = {t: s for t, s in no_retrieved.items() if t in relevant_topics}
        for q in relevant_topics.difference(set(no_retrieved.keys())):
            no_retrieved.update({q: 0.0})

        for measure, _ in measures:
            run = extract_run(raw_run, measure)
            # add missing scores
            for q in relevant_topics.difference(set(run.keys())):
                run.update({q: 0.0})

            # only evaluate relevant topics and skip the others
            run = {t: s for t, s in run.items() if t in relevant_topics}

            score = round(sum(run.values()) / len(run.keys()), 2)

            max_m[measure] = max(max_m[measure], score)
            s_row[measure] = score
        score_rows[name] = s_row
    return score_rows, max_m


def generate_table(measures: dict, strategy: str, first_stages: List[str], benchmark: Benchmark, results: dict = None,
                   min_docs_per_topic: int = None, min_recall: float = 0.0):
    score_rows_fs = dict()
    measures = [(k, v) for k, v in measures.items()]
    for first_stage in first_stages:
        print("--" * 60)
        print(
            f"Generating score table for FirstStage {first_stage} with benchmark {benchmark.name} and strategy {strategy}")
        path = os.path.join(EVAL_DIR, f"{first_stage}_{strategy}_", f"{benchmark.name}")
        if not results:
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

        score_rows_fs[first_stage] = calculate_table_data(measures, results, relevant_topics)
        results = None

    print('==' * 60)
    print(f'Evaluation based on {len(relevant_topics)} topics')
    print('==' * 60)
    print("--" * 60)
    print("Creating table content")
    print("--" * 60)
    # create tabular LaTeX code
    rows = []
    rows.append("% begin autogenerated")
    rows.append("\\toprule")
    rows.append(" & ".join(["Ranking Method", *(str(m[1]) for m in measures)]) + " \\\\")

    for first_stage in first_stages:
        score_rows, max_m = score_rows_fs[first_stage]
        rows.append("\\midrule")
        rows.append(f"\\multicolumn{{8}}{{c}}{{\\textbf{{{first_stage}}}}} \\\\")
        rows.append("\\midrule")

        for name, scores in score_rows.items():
            method_name = name.replace('DocumentRanker', '').split('-')[0].replace('_', ' ')
            if REPORT_JUST_METHODS_IN_PAPER:
                if method_name not in METHODS_TO_REPORT_IN_PAPER:
                    continue
                method_name = METHODS_TO_REPORT_IN_PAPER[method_name]

            row = f"{method_name} & "
            row += " & ".join(f"\\textbf{{{str(s)}}}" if max_m[m] == s else str(s) for m, s in scores.items())
            row += " \\\\"
            rows.append(row)

    rows.append("\\bottomrule")
    rows.append("% end autogenerated")

    print("\n".join(rows))
    print("--" * 60)


def load_document_count_statistics(file_path) -> List[tuple]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Statistics file {file_path} not found")
    with open(file_path, "rt") as file:
        data = json.load(file)["data"]
        doc_count = dict()
        for query in data:
            topic = query["topic"].replace("<", "").split()[0].strip()
            documents = query["documents"]
            doc_count[topic] = documents
        return sorted(doc_count.items(), key=lambda x: x[0])


def load_results(path):
    file_path = os.path.join(path, "results.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError("Create results.json first by calling evaluate_runs()")
    with open(file_path, "rt") as file:
        return json.load(file)


def save_results(results: dict, strategy: str, first_stage: str, benchmark: str):
    path = os.path.join(EVAL_DIR, f"{first_stage}_{strategy}_", f"{benchmark}")
    print(f"Save results to {path}/results.json")
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "results.json"), "wt") as file:
        json.dump(results, file, indent=2)


def evaluate_runs(with_bm25_native=False, benchmarks=BENCHMARKS):
    print("Starting evaluation")
    for first_stage in FIRST_STAGE_NAMES:
        for strategy in CONCEPT_STRATEGIES:
            run_path_root = os.path.join(RESULT_DIR, f"{first_stage}_{strategy}")
            for benchmark in benchmarks:
                qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
                qrel = load_qrel_from_file(str(qrel_path))

                evaluator = pe.RelevanceEvaluator(qrel, MEASURES, judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)
                ranker_results = dict()

                run_path_pattern = os.path.join(run_path_root, f"{benchmark.name}_*.txt")
                run_paths = glob.glob(run_path_pattern)

                for run_path in run_paths:
                    run = load_run_from_file(run_path)
                    ranker_name = run_path.strip().rsplit(".")[0].split("/")[-1].split("_", 1)[-1]
                    ranker_results[ranker_name] = evaluator.evaluate(run)

                if with_bm25_native:
                    bm25_run_path = os.path.join(RESULT_DIR_BASELINES, f'{benchmark.name}_BM25.txt')
                    bm25_run = load_run_from_file(bm25_run_path)
                    ranker_results["BM25 Native"] = evaluator.evaluate(bm25_run)

                save_results(ranker_results, strategy, first_stage, benchmark.name)
                # generate_diagram(MEASURES, strategy, first_stage, benchmark.name, ranker_results)
                # generate_table(RESULT_MEASURES, strategy, [first_stage], benchmark, ranker_results)


def evaluate_first_stage_runs(measures):
    measures = [(k, v) for k, v in measures.items()]

    print("Starting evaluation")
    for benchmark in BENCHMARKS:
        print('--' * 60)
        print(f'First stage evaluation for {benchmark.name} with concept strategies: {CONCEPT_STRATEGIES}')
        print('--' * 60)
        print("--" * 60)
        print("Creating table content")
        print("--" * 60)
        for concept_strat in CONCEPT_STRATEGIES:
            fs_results = dict()

            last_first_stage = None
            for first_stage in FIRST_STAGE_NAMES:
                last_first_stage = first_stage
                qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
                qrel = load_qrel_from_file(str(qrel_path))

                evaluator = pe.RelevanceEvaluator(qrel, MEASURES, judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

                run_path = os.path.join(RESULT_DIR_FIRST_STAGE, f'{benchmark.name}_{first_stage}_{concept_strat}.txt')
                run = load_run_from_file(run_path)

                fs_results[first_stage] = evaluator.evaluate(run)

            skipped_topics = SKIPPED_TOPICS[benchmark.name].copy()
            if EVALUATION_SKIP_BAD_TOPICS:
                # It is enough to load the last first stage because the statistics concerning translation and components
                # is consistent across a single concept strategy for a benchmark
                stats_dir = os.path.join(RESULT_DIR_FIRST_STAGE, 'statistics')
                stats_path = os.path.join(stats_dir, f'{benchmark.name}_{last_first_stage}_{concept_strat}.json')
                with open(stats_path, 'rt') as f:
                    stats = json.load(f)

                for topic_id in stats:
                    if 'skipped' in stats[topic_id]:
                        skipped_topics.append(topic_id)

            relevant_topics = {str(q.query_id) for q in benchmark.topics
                               if str(q.query_id) not in skipped_topics}
            print(f'Considered {len(relevant_topics)} relevant topics')
            results = [(k, v) for k, v in sorted(fs_results.items(), key=lambda x: x[0])]

            score_rows, max_m = calculate_table_data(measures, results, relevant_topics)

            # create tabular LaTeX code
            rows = []
            rows.append("% begin autogenerated")
            rows.append('\\multicolumn{8}{c}{\\textbf{' + concept_strat + '}} \\\\')
            rows.append("\\toprule")
            rows.append(" & ".join(["First Stage", *(str(m[1]) for m in measures)]) + " \\\\")

            for name, scores in score_rows.items():
                row = f"{name.replace('FirstStage', '').split('-')[0].replace('_', ' ')} & "
                row += " & ".join(
                    f"\\textbf{{{str(s)}}}" if max_m[m] == s else str(s) for m, s in scores.items())
                row += " \\\\"
                rows.append(row)

            rows.append("\\bottomrule")
            rows.append("% end autogenerated")

            print("\n".join(rows))

        print("--" * 60)
        print('\n\n\n')


if __name__ == "__main__":
    evaluate_first_stage_runs(FS_RESULT_MEASURES)
    print('\n\n\n')
    evaluate_runs()
    for benchmark in BENCHMARKS:
        for strategy in CONCEPT_STRATEGIES:
            print("--" * 60)
            print(f'Benchmark:      : {benchmark.name}')
            print(f'Concept strategy: {strategy}')
            print("--" * 60)
            generate_table(RESULT_MEASURES, strategy, FIRST_STAGE_NAMES, benchmark)
            # generate_table(RESULT_MEASURES, strategy, FIRST_STAGE_NAMES, benchmark, min_docs_per_topic=100)
            # generate_table(RESULT_MEASURES, strategy, FIRST_STAGE_NAMES, benchmark, min_recall=0.2)
    # generate_diagram(RESULT_MEASURES, CONCEPT_STRATEGIES[0], FIRST_STAGES, BENCHMARKS[0])
