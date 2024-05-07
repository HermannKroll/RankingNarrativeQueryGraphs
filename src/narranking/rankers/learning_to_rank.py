import datetime
import json
import os.path
import random

import numpy as np
import pandas as pd
import pyterrier as pt
import pytrec_eval as pe
import tqdm
from sklearn.ensemble import RandomForestRegressor

from narraplay.documentranking.analysis.util import load_run_from_file, load_qrel_from_file
from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import RESULT_DIR, RUNS_DIR, QRELS_PATH, RESULT_DIR_FIRST_STAGE, RESULT_DIR_LTR
from narraplay.documentranking.run_config import RANKER_BASE_MIN, BENCHMARKS, FIRST_STAGES, CONCEPT_STRATEGIES, \
    JUDGED_DOCS_ONLY_FLAG


def df_from_run_path(run_path: str) -> pd.DataFrame:
    run: dict = load_run_from_file(run_path)
    data = list()
    for qid, doc_scores in run.items():
        for docno, score in doc_scores.items():
            data.append((str(qid), str(docno), float(score)))
    data.sort(key=lambda x: (x[0], x[1]))
    return pd.DataFrame(data, columns=["qid", "docno", "score"])


def df_from_qrel_path(run_path: str) -> pd.DataFrame:
    run: dict = load_qrel_from_file(run_path)
    data = list()
    for qid, doc_labels in run.items():
        for docno, label in doc_labels.items():
            data.append((str(qid), str(docno), int(label)))
    data.sort(key=lambda x: (x[0], x[1]))
    return pd.DataFrame(data, columns=["qid", "docno", "label"])


def df_from_topics(benchmark: Benchmark):
    data = list()
    for topic in benchmark.topics:
        data.append((str(topic.query_id), topic.get_benchmark_string()))
    return pd.DataFrame(data, columns=["qid", "query"])


# custom base transformer lambda function
def by_query(doc, base_dataframe: pd.DataFrame):
    # doc ("qid", value)
    # return dataframe for the corresponding qid
    return base_dataframe.loc[(base_dataframe["qid"].isin(doc["qid"]))].copy()


# custom feature transformer lambda function
def doc_score(doc, feature_df: pd.DataFrame):
    # doc ("qid", value) ("docno", value) ("score", value) ("rank", value)
    # return df of floats for each document with the corresponding qid and docno
    score = feature_df[(feature_df["qid"].isin(doc["qid"])) & (feature_df["docno"].isin(doc["docno"]))]["score"]
    return score


def main(criterion):
    """
    Learning to Rank approach for different ranking strategies using pyterrier
    - https://pyterrier.readthedocs.io/en/latest/apply.html#pyterrier-apply
    - https://pyterrier.readthedocs.io/en/latest/ltr.html
    """

    start = datetime.datetime.now()

    benchmark = BENCHMARKS[0]
    first_stage = FIRST_STAGES[0].name
    strategy = CONCEPT_STRATEGIES[0]

    topic_base = [str(topic.query_id) for topic in benchmark.topics]

    qrels = df_from_qrel_path(str(os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])))
    topics = df_from_topics(benchmark)

    assert sorted(topic_base) == sorted(topics["qid"].to_list())

    base_path = os.path.join(RESULT_DIR_FIRST_STAGE, f"{benchmark.name}_{first_stage}_{strategy}.txt")
    base_df = df_from_run_path(base_path)

    transformer_base = pt.apply.by_query(lambda doc: by_query(doc, base_df), batch_size=1)

    MAX_ROUNDS = 100
    skipped = 0
    score_results = dict()
    for _ in tqdm.tqdm(range(MAX_ROUNDS)):
        # shuffle the topics and split them into half (train <= test)
        random.shuffle(topic_base)
        train_topic_base, test_topic_base = np.split(topic_base, [int(0.49 * len(topic_base))])

        train_topic_base.sort()
        test_topic_base.sort()

        round_key = "|".join(train_topic_base)
        if round_key in score_results:
            skipped += 1
            continue

        train_topics = topics.loc[topics["qid"].isin(train_topic_base)].copy()
        test_topics = topics.loc[topics["qid"].isin(test_topic_base)].copy()

        train_qrels = qrels.loc[qrels["qid"].isin(train_topic_base)].copy()
        test_qrels = qrels.loc[qrels["qid"].isin(test_topic_base)].copy()

        features = None
        for ranker_name in RANKER_BASE_MIN:
            run_path = os.path.join(RESULT_DIR, f"{first_stage}_{strategy}", f"{benchmark.name}_{ranker_name}.txt")
            run_df = df_from_run_path(run_path)

            assert len(run_df) == len(base_df)
            assert sorted(run_df["docno"].to_list()) == sorted(base_df["docno"].to_list())

            transformer = pt.apply.doc_score(lambda doc: doc_score(doc, run_df), batch_size=1)

            if features is None:
                features = transformer
            else:
                features = features ** transformer

        pipeline = transformer_base >> features

        # apply LTR (default form regression, other: ltr, fastrank)
        rf = RandomForestRegressor(n_estimators=400, criterion=criterion)
        rf_pipe: pt.pipelines.Pipeline = pipeline >> pt.ltr.apply_learned_model(rf)
        rf_pipe.fit(train_topics, train_qrels, test_topics, test_qrels)

        results: pd.DataFrame = pt.pipelines.Experiment(
            [rf_pipe],
            test_topics,
            qrels,
            ["P_20", ],
            names=["LTR_Pipeline"]
        )

        relevant_score = round(results["P_20"].iloc[0], 2)
        score_results[round_key] = relevant_score
        print(round_key, relevant_score)

    print("Skipped", skipped, "shuffling rounds.")
    print("Saving results...")
    file_name = os.path.join(RESULT_DIR_LTR, f"{first_stage}_{strategy}_{benchmark.name}_ltr_{criterion}.json")
    with open(file_name, "wt") as fd:
        json.dump(score_results, fd, indent=2)
    print("Finished after", datetime.datetime.now() - start)


def evaluate_results(criterion: str):
    benchmark = BENCHMARKS[0]
    first_stage = FIRST_STAGES[0].name
    strategy = CONCEPT_STRATEGIES[0]

    file_name = os.path.join(RESULT_DIR_LTR, f"{first_stage}_{strategy}_{benchmark.name}_ltr_{criterion}.json")
    if not os.path.isfile(file_name):
        raise FileNotFoundError("The combination of parameters is not evaluated yet.", file_name)
    with open(file_name, "r") as fd:
        json_data = json.load(fd)

    max_ltr_score = 0.0
    max_ltr_combo = None
    for combo, score in json_data.items():
        if score > max_ltr_score:
            max_ltr_score = score
            max_ltr_combo = combo

    topics = [str(t.query_id) for t in benchmark.topics]
    relevant_topics = set(topics) - set(max_ltr_combo.split("|"))

    # load run file for comparison; cut off the qids, that are part of the LTR training-set
    path = os.path.join(RESULT_DIR, f"{first_stage}_{strategy}",
                        f"{benchmark.name}_WeightedDocumentRanker_min-0.2-0.2-0.2-0.2-0.2.txt")
    run = load_run_from_file(path)
    run = {k: v for k, v in run.items() if k in relevant_topics}

    assert set(relevant_topics) == set(run.keys())

    qrel = load_qrel_from_file(str(os.path.join(RUNS_DIR, QRELS_PATH[benchmark.name])))
    evaluator = pe.RelevanceEvaluator(qrel, ["P_20"], judged_docs_only_flag=JUDGED_DOCS_ONLY_FLAG)
    result = evaluator.evaluate(run)
    graph_score = sum(s["P_20"] for s in result.values()) / len(relevant_topics)

    print("regression criterion  :", criterion)
    print("max LTR score combo   :", max_ltr_combo)
    print("relevant ltr topics   :", relevant_topics)
    print("Learning To Rank Score:", round(max_ltr_score, 2))
    print("Graph Reranking Score :", round(graph_score, 2))
    print("==" * 60)


if __name__ == "__main__":
    regression_error_min_criteria = ["poisson", "squared_error"]

    for c in regression_error_min_criteria:
        # main(c)
        evaluate_results(c)
