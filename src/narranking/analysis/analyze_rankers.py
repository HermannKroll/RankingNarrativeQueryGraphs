import json
import os

import pandas as pd
import pytrec_eval as pe

from narraplay.documentranking.analysis.util import load_qrel_from_file, load_run_from_file, extract_run, \
    draw_bar_chart, PLT_VARIANT_COLOR_CYCLE
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RUNS_DIR, QRELS_PATH, RESULT_DIR, RESULT_DIR_BASELINES, \
    RESULT_DIR_FIRST_STAGE
from narraplay.documentranking.run_config import BENCHMARKS, FIRST_STAGES, JUDGED_DOCS_ONLY_FLAG, \
    SKIPPED_TOPICS, EVALUATION_SKIP_BAD_TOPICS, FIRST_STAGE_TO_PRINT_NAME

MEASURES = {
    'recall_1000': 'Recall@1000',
    'ndcg_cut_10': 'nDCG@20',
    'ndcg_cut_20': 'nDCG@20',
    'P_10': 'P@20',
    'P_20': 'P@20',
}

RANKERS_TO_COMPARE = [
    "WeightedDocumentRanker_min-0.25-0.25-0.25-0.25",
]

RANKER_TO_PRINT_NAME = {
    "WeightedDocumentRanker": "GraphRank",
    "BM25Text": "Rerank BM25",
    "BM25": "Native BM25",
}

SUBDIR_PATH = "rankers"


def generate_diagram(measures: list, benchmark: Benchmark, results: dict = None):
    for measure in measures:
        dfs = list()
        for config, runs in results.items():
            extracted_runs = list()
            prefix = FIRST_STAGE_TO_PRINT_NAME[config]
            for name, run in runs:
                strat, name = name.split("/")
                name = RANKER_TO_PRINT_NAME[name.split("_", 1)[-1].split("_")[0].split(".")[0]]

                if name != "Native BM25" and "ontology" in strat:
                    name = prefix + " + Ontology + " + name
                elif name != "Native BM25":
                    name = prefix + " + " + name

                run = extract_run(run, set(), measure)
                run = pd.DataFrame.from_dict(run, orient="index", columns=[name])
                extracted_runs.append(run)

            df = pd.concat(extracted_runs, axis="columns")
            dfs.append(df)
            if len(runs) == 2:
                print("use alternative color cycle")
                draw_bar_chart(dfs, SUBDIR_PATH, measure, benchmark.name, config, color_cycle=PLT_VARIANT_COLOR_CYCLE)
            else:
                draw_bar_chart(dfs, SUBDIR_PATH, measure, benchmark.name, config)


def main():
    for benchmark in BENCHMARKS:
        qrel_path = os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])
        qrel = load_qrel_from_file(str(qrel_path))

        evaluator = pe.RelevanceEvaluator(qrel, [*MEASURES.keys()], judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)

        for first_stage in FIRST_STAGES:
            results = dict()
            base_path_like = os.path.join(RESULT_DIR, f"{first_stage.name}_likesimilarity")
            base_path_like_onto = os.path.join(RESULT_DIR, f"{first_stage.name}_likesimilarityontology")
            run_paths = [os.path.join(base_path_like, f"{benchmark.name}_{RANKERS_TO_COMPARE[0]}.txt"),
                         os.path.join(base_path_like_onto, f"{benchmark.name}_{RANKERS_TO_COMPARE[0]}.txt"),
                         os.path.join(RESULT_DIR_BASELINES, f"{benchmark.name}_BM25.txt")]

            skipped_topics = SKIPPED_TOPICS[benchmark.name].copy()

            if EVALUATION_SKIP_BAD_TOPICS:
                # Load statistics concerning translation and components
                stats_dir = os.path.join(RESULT_DIR_FIRST_STAGE, 'statistics')
                stats_path = os.path.join(stats_dir, f'{benchmark.name}_{first_stage.name}_likesimilarity.json')
                with open(stats_path, 'rt') as f:
                    stats = json.load(f)

                for topic_id in stats:
                    if 'skipped' in stats[topic_id]:
                        skipped_topics.append(topic_id)

            print(benchmark.name, first_stage.name, "Compare following runs")
            for run_path in run_paths:
                if not os.path.exists(run_path):
                    continue
                print(run_path)
                run = load_run_from_file(run_path)
                # filter bad topics
                run = {k: v for k, v in run.items() if k not in skipped_topics}
                evaluated = evaluator.evaluate(run)
                if first_stage.name not in results:
                    results[first_stage.name] = list()

                results[first_stage.name].append((run_path.split("results/")[-1], evaluated))

            generate_diagram(list(MEASURES.keys()), benchmark, results)


if __name__ == '__main__':
    main()
