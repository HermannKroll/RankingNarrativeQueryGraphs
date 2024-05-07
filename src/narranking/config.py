import os

import narraplay.config

BM25_RANKED_DOCUMENT_CUTOFF = 1000

ROOT_DIR = "/ssd2/kroll/retrieval/"

DATA_DIR = os.path.join(ROOT_DIR, "data")
RESOURCE_DIR = narraplay.config.RESOURCE_DIR

PUBMED_BASELINE_ID_DIR = os.path.join(DATA_DIR, "pubmed_baselines")

PM2020_TOPIC_FILE = os.path.join(RESOURCE_DIR, "benchmarks/trec_pm2020_topics.xml")
PM2020_MAMMELS_TAX_ID_FILE = os.path.join(RESOURCE_DIR, "benchmarks/pm2020_mammels_tax_ids.txt")

TRIP_CLICK_TOPIC_FILES = [os.path.join(RESOURCE_DIR, "benchmarks/tripclick_topics.head.test.xml"),
                          os.path.join(RESOURCE_DIR, "benchmarks/tripclick_topics.head.val.xml")]
TRIP_JUDGE_TOPIC_FILES = [os.path.join(RESOURCE_DIR, "benchmarks/tripjudge_topics.head.test.xml")]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DIAGRAMS_DIR = os.path.join(ROOT_DIR, "diagrams")
EVAL_DIR = os.path.join(ROOT_DIR, "evaluation")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")
RESULT_DIR = os.path.join(ROOT_DIR, 'results')
RESULT_DIR_TOPICS = os.path.join(RESULT_DIR, "topics")
if not os.path.exists(RESULT_DIR_TOPICS):
    os.makedirs(RESULT_DIR_TOPICS)

RESULT_DIR_FIRST_STAGE = os.path.join(RESULT_DIR, "FirstStage")
RESULT_DIR_FIST_STAGE_BASELINES = os.path.join(RESULT_DIR, "FirstStageBaselines")
RESULT_DIR_BASELINES = os.path.join(RESULT_DIR, "Baselines")
if not os.path.exists(RESULT_DIR_BASELINES):
    os.makedirs(RESULT_DIR_BASELINES)

RESULT_DIR_LTR = os.path.join(RESULT_DIR, "LearningToRank")
RESULT_DIR_HYPERPARAMS = os.path.join(RESULT_DIR, "HyperparameterSearch")

PYTERRIER_INDEX_PATH = os.path.join(DATA_DIR, "pyterrier_indexes")
if not os.path.exists(PYTERRIER_INDEX_PATH):
    os.makedirs(PYTERRIER_INDEX_PATH)

DOCUMENT_TEXT_INDEX_PATH = os.path.join(PYTERRIER_INDEX_PATH, "Document_all_text")

if not os.path.exists(DIAGRAMS_DIR):
    os.makedirs(DIAGRAMS_DIR)
if not os.path.exists(EVAL_DIR):
    os.makedirs(EVAL_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(RESULT_DIR_FIRST_STAGE):
    os.makedirs(RESULT_DIR_FIRST_STAGE)
if not os.path.exists(RESULT_DIR_FIST_STAGE_BASELINES):
    os.makedirs(RESULT_DIR_FIST_STAGE_BASELINES)

QUERY_YIELD_PER_K = 1000000

QRELS_PATH = {
    "trec-pm-2017-abstracts": "trec-pm-2017-abstracts/qrels-final-abstracts.txt",
    "trec-pm-2018-abstracts": "trec-pm-2018-abstracts/qrels-treceval-abstracts-2018-v2.txt",
    "trec-pm-2019-abstracts": "trec-pm-2019-abstracts/qrels-treceval-abstracts.2019.txt",
    "trec-pm-2020-abstracts": "trec-pm-2020/qrels-reduced-phase1-treceval-2020.txt",
    "trec-pm-2017-cds": "trec-pm-2017-cds/qrels-final-trials.txt",
    "trec-pm-2018-cds": "trec-pm-2018-cds/qrels-treceval-clinical_trials-2018-v2.txt",
    "trec-pm-2019-cds": "trec-pm-2019-cds/qrels-treceval-trials.38.txt",
    "trec-covid-rnd5-abstracts": "trec-covid-rnd5/qrels-covid_d5_j0.5-5.txt",
    "trec-covid-rnd5-fulltexts": "trec-covid-rnd5/qrels-covid_d5_j0.5-5.txt",
    "trip-click": "trip_click/qrels_all.txt",
    # ["trip_click/qrels.raw.head.test.txt", "trip_click/qrels.raw.head.val.txt"],
    "trip-judge": "trip_judge/qrels_2class.txt"
}
DATASET_TO_PUBMED_BASE_ID_FILE = {
    "trec-pm-2017-abstracts": "pubmed_baseline_pm17.txt",
    "trec-pm-2018-abstracts": "pubmed_baseline_pm17.txt",
    "trec-pm-2019-abstracts": "pubmed_baseline_pm19.txt",
    "trec-pm-2020-abstracts": "pubmed_baseline_pm19.txt"
}

MINIMUM_TRANSLATION_THRESHOLD = 0.9
