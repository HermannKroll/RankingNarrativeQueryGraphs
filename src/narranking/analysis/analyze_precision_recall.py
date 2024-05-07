import json
import os
from collections import defaultdict

import numpy as np
import pytrec_eval as pe
from matplotlib import pyplot as plt
from tqdm import tqdm

from narraplay.documentranking.config import RUNS_DIR, DIAGRAMS_DIR, QRELS_PATH
from narraplay.documentranking.entity_tagger_like import EntityTaggerLike
from narraplay.documentranking.evaluate import load_qrel_from_file, load_run_from_file
from narraplay.documentranking.query import AnalyzedQuery
from narraplay.documentranking.run_config import BENCHMARKS, FIRST_STAGES, SKIPPED_TOPICS, \
    CONCEPT_STRATEGIES, FIRST_STAGE_CUTOFF, JUDGED_DOCS_ONLY_FLAG

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METRICS = ["set_P", "set_recall"]
DISTANCE_MEASURES = [
    "levenshtein",
    # "jarowinkler"
]

SUBDIR_PATH = os.path.join(DIAGRAMS_DIR, "precision_recall")
SUBDIR_PATH_TMP = os.path.join(SUBDIR_PATH, "tmp")


def scores_mean(eval_result: dict, relevant_topics: set, metrics: list):
    print("Evaluated topics:", len(eval_result.keys()))
    print("Relevant topics :", len(relevant_topics))
    print("Missing topics  :", relevant_topics - set(eval_result.keys()))
    print("==" * 60)

    num_topics = len(relevant_topics)
    metric_scores = {metric: 0.0 for metric in metrics}

    for topic, metric_results in eval_result.items():
        for metric, score in metric_results.items():
            metric_scores[metric] += score

    for metric in metric_scores:
        metric_scores[metric] = metric_scores[metric] / float(num_topics)

    return metric_scores


def precision_recall_analysis(judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG):
    jdof = str(judged_docs_only_flag).lower()
    for benchmark in BENCHMARKS:
        qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
        qrel = load_qrel_from_file(qrel_path)
        evaluator = pe.RelevanceEvaluator(qrel, METRICS, judged_docs_only_flag=judged_docs_only_flag)

        relevant_topics = {str(q) for q in range(1, len(benchmark.topics) + 1) if
                           str(q) not in SKIPPED_TOPICS[benchmark.name]}

        for first_stage in FIRST_STAGES:
            for concept_strategy in CONCEPT_STRATEGIES:
                dm_results = dict()
                for distance_measure in DISTANCE_MEASURES:
                    th_results = dict()
                    for threshold in THRESHOLDS:
                        file_name = (f'{benchmark.name}_{first_stage.name}_{concept_strategy}'
                                     f'_{distance_measure}_{threshold}.txt')
                        run_path = os.path.join(SUBDIR_PATH_TMP, file_name)
                        run = load_run_from_file(run_path)

                        eval_result = evaluator.evaluate(run)
                        score = scores_mean(eval_result, relevant_topics, metrics=METRICS)
                        th_results[str(threshold)] = score

                    dm_results[distance_measure] = th_results

                out_file_name = f'{benchmark.name}_{first_stage.name}_{concept_strategy}_pr_analysis_{jdof}.json'
                with open(os.path.join(SUBDIR_PATH, out_file_name), "wt") as out_file:
                    json.dump(dm_results, out_file, indent=2)


def precision_recall_graph(judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG):
    COLORS = ['green', 'red']
    MARKERS = ['+', 'x']
    LINES = ['-', '--']

    for benchmark in BENCHMARKS:
        for first_stage in FIRST_STAGES:
            for concept_strategy in CONCEPT_STRATEGIES:
                jdof = str(judged_docs_only_flag).lower()
                result_file = f'{benchmark.name}_{first_stage.name}_{concept_strategy}_pr_analysis_{jdof}.json'
                with open(os.path.join(SUBDIR_PATH, result_file), "rt") as file:
                    results = json.load(file)

                # plt.xlim([0, 1])
                # plt.ylim([0, 1])

                assert len(DISTANCE_MEASURES) <= len(MARKERS)

                for i, distance_measure in enumerate(DISTANCE_MEASURES):
                    values = list()
                    xy_tags = list()

                    score2threshold = defaultdict(list)
                    for key, scores in results[distance_measure].items():
                        y, x = scores[METRICS[0]], scores[METRICS[1]]
                        score2threshold[(x, y)].append(float(key))

                    for score, thresholds in score2threshold.items():
                        values.append(score)
                        xy_tag = f"{min(thresholds)}-{max(thresholds)}" if len(thresholds) > 1 else str(thresholds[0])
                        xy_tags.append(xy_tag)

                    x, y = np.array(values).T
                    x_center = (max(x) + min(x)) / 2

                    # draw scatterplot with threshold marks
                    plt.scatter(x, y, marker=MARKERS[i], color=COLORS[i])
                    for xy_tag, px, py in zip(xy_tags, x, y):
                        if px < x_center:
                            plt.annotate(xy_tag, xy=(px, py),
                                         xytext=(px + 0.0005, py),
                                         ha='left', va='center')
                        else:
                            plt.annotate(xy_tag, xy=(px, py),
                                         xytext=(px - 0.0005, py),
                                         ha='right', va='center')

                    # polynomial (deg2) regression
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    x_p = np.linspace(min(x), max(x), 100)
                    plt.plot(x_p, p(x_p), LINES[i], color=COLORS[i], label=distance_measure.title())

                plt.ylabel('Precision')
                plt.xlabel('Recall')
                plt.legend()
                # plt.title(f"Precision-Recall Analysis\n{benchmark.name} + {first_stage.name}")
                fig_path = f'{benchmark.name}_{first_stage.name}_{concept_strategy}_{jdof}_fig.png'
                plt.savefig(os.path.join(SUBDIR_PATH, fig_path))
                plt.close()


def generate_first_stage_run_files():
    tagger_v2 = EntityTaggerLike.instance()
    os.makedirs(SUBDIR_PATH_TMP, exist_ok=True)
    for stage in FIRST_STAGES:
        for bench in BENCHMARKS:
            print('==' * 60)
            print('==' * 60)
            print(f'Running configuration')
            print(f'- FIRST STAGE: {stage.name}')
            print(f'- BENCHMARK: {bench}')
            print('==' * 60)
            print('==' * 60)

            for concept_strategy in CONCEPT_STRATEGIES:
                for distance_measure in DISTANCE_MEASURES:
                    tagger_v2.set_string_distance_measure(distance_measure)
                    for threshold in tqdm(THRESHOLDS,
                                          desc=f'{stage.name} {bench.name} {concept_strategy} {distance_measure}'):
                        tagger_v2.set_min_sim_concept_translation_threshold(threshold)

                        file_name = f'{bench.name}_{stage.name}_{concept_strategy}_{distance_measure}_{threshold}.txt'
                        path = os.path.join(SUBDIR_PATH_TMP, file_name)
                        if os.path.isfile(path):
                            print(f'Skipping already computed configuration: {path}')
                            continue

                        print('==' * 60)
                        print('==' * 60)
                        print(f'- Distance Measure: {distance_measure}')
                        print(f'- Threshold: {threshold}')
                        print('==' * 60)
                        print('==' * 60)

                        result_lines = []
                        for idx, q in enumerate(bench.topics):
                            print('--' * 60)
                            print(f'Evaluating query {q}')
                            print(f'Query components: {list(q.get_query_components())}')
                            analyzed_query = AnalyzedQuery(q, concept_strategy=concept_strategy)

                            ranked_docs = []
                            for collection in bench.document_collections:
                                docs_for_c, _, _ = stage.retrieve_documents(analyzed_query, collection, bench)
                                ranked_docs.extend(docs_for_c)

                            print(f'Received {len(ranked_docs)} documents')
                            ranked_docs.sort(key=lambda x: (x[1], x[0]), reverse=True)
                            if FIRST_STAGE_CUTOFF > 0:
                                ranked_docs = ranked_docs[:FIRST_STAGE_CUTOFF]
                                print(f'First stage cutoff applied -> {len(ranked_docs)} remaining')

                            for rank, (did, score) in enumerate(ranked_docs):
                                result_line = f'{q.query_id}\tQ0\t{did}\t{rank + 1}\t{score}\t{stage.name}'
                                result_lines.append(result_line)

                        print(f'Write ranked results to {path}')
                        with open(path, 'wt') as f:
                            f.write('\n'.join(result_lines))


if __name__ == "__main__":
    generate_first_stage_run_files()

    precision_recall_analysis(judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)
    precision_recall_graph(judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

    precision_recall_analysis(judged_docs_only_flag=not JUDGED_DOCS_ONLY_FLAG)
    precision_recall_graph(judged_docs_only_flag=not JUDGED_DOCS_ONLY_FLAG)
