import os.path

MIN_SCORE_THRESHOLD = 0.6
RANKED_DOCUMENT_CUTOFF = 1000

DATA_DIR = "/home/kroll/NarrativeIntelligence/lib/NarrativePlayground/data/"
RESOURCE_DIR = "/home/kroll/NarrativeIntelligence/lib/NarrativePlayground/resources"

PUBMED_BASELINE_ID_DIR = os.path.join(DATA_DIR, "pubmed_baselines")

PM2020_TOPIC_FILE = os.path.join(RESOURCE_DIR, "benchmarks/trec_pm2020_topics.xml")
CONCEPT_INDEX_PATH = "/home/kroll/tpdl2023_graph_concept_index.pkl"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

RESULT_DIR = os.path.join(DATA_DIR, 'results')
RESULT_DIR_FIRST_STAGE = os.path.join(RESULT_DIR, "FirstStage")
RESULT_DIR_TEMP = os.path.join(RESULT_DIR, 'tmp')

PYTERRIER_INDEX_PATH = os.path.join(DATA_DIR, "pyterrier_indexes")
if not os.path.exists(PYTERRIER_INDEX_PATH):
    os.makedirs(PYTERRIER_INDEX_PATH)

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(RESULT_DIR_FIRST_STAGE):
    os.makedirs(RESULT_DIR_FIRST_STAGE)
if not os.path.exists(RESULT_DIR_TEMP):
    os.makedirs(RESULT_DIR_TEMP)

IGNORE_DEMOGRAPHIC = True

QUERY_YIELD_PER_K = 1000000