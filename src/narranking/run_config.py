from narraplay.documentranking.benchmark import Benchmark
from narraplay.documentranking.config import MINIMUM_TRANSLATION_THRESHOLD
from narraplay.documentranking.entity_tagger_like import CONCEPT_TRANSLATION_SIMILARITY
from narraplay.documentranking.first_stages.first_stage_graph import FirstStageGraphRetriever
from narraplay.documentranking.first_stages.first_stage_partial_graph import FirstStagePartialGraphRetriever
from narraplay.documentranking.rankers.ranker_b25_text import BM25Text
from narraplay.documentranking.rankers.ranker_confidence import ConfidenceDocumentRanker
from narraplay.documentranking.rankers.ranker_confidence_avg import ConfidenceAvgDocumentRanker
from narraplay.documentranking.rankers.ranker_confidence_max import ConfidenceMaxDocumentRanker
from narraplay.documentranking.rankers.ranker_connectivity import ConnectivityDocumentRanker
from narraplay.documentranking.rankers.ranker_connectivity_inverse import ConnectivityInverseDocumentRanker
from narraplay.documentranking.rankers.ranker_connectivity_normalized import ConnectivityNormalizedDocumentRanker
from narraplay.documentranking.rankers.ranker_coverage import ConceptCoverageDocumentRanker
from narraplay.documentranking.rankers.ranker_document_length import DocLengthDocumentRanker
from narraplay.documentranking.rankers.ranker_equal import EqualDocumentRanker
from narraplay.documentranking.rankers.ranker_idf_avg import IDFAvgDocumentRanker
from narraplay.documentranking.rankers.ranker_idf_max import IDFMaxDocumentRanker
from narraplay.documentranking.rankers.ranker_idf_min import IDFMinDocumentRanker
from narraplay.documentranking.rankers.ranker_idf_sum import IDFSumDocumentRanker
from narraplay.documentranking.rankers.ranker_position import ConceptPositionDocumentRanker
from narraplay.documentranking.rankers.ranker_relational_sim import RelationalSimDocumentRanker
from narraplay.documentranking.rankers.ranker_relational_sim_normalized import RelationalSimNormalizedDocumentRanker
from narraplay.documentranking.rankers.ranker_relational_sim_tfidf import RelationalSimTFIDFDocumentRanker
from narraplay.documentranking.rankers.ranker_relational_sim_translation import RelationalSimTranslationDocumentRanker
from narraplay.documentranking.rankers.ranker_sentence_weight import SentenceWeightRanker
from narraplay.documentranking.rankers.ranker_tf_avg import TfAvgDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_idf_avg import TfIdfAvgDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_idf_max import TfIdfMaxDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_idf_min import TfIdfMinDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_idf_only_concepts_max import TfIdfOnlyConceptsMaxDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_idf_plus_concepts_max import TfIdfPlusConceptsMaxDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_max import TfMaxDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_min import TfMinDocumentRanker
from narraplay.documentranking.rankers.ranker_tf_sum import TfSumDocumentRanker
from narraplay.documentranking.rankers.ranker_translation import TranslationDocumentRanker

BENCHMARKS = [
    Benchmark("trec-pm-2020-abstracts", "", ["PubMed"], load_from_file=True),
    Benchmark("trec-pm-2017-abstracts", "medline/2017/trec-pm-2017", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2017-cds", "clinicaltrials/2017/trec-pm-2017", ["trec-pm-2017-cds"]),

    Benchmark("trec-pm-2018-abstracts", "medline/2017/trec-pm-2018", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2018-cds", "clinicaltrials/2017/trec-pm-2018", ["trec-pm-2018-cds"]),

    Benchmark("trec-pm-2019-abstracts", "clinicaltrials/2019/trec-pm-2019", ["PubMed", "trec-pm-201X-extra"]),
    Benchmark("trec-pm-2019-cds", "clinicaltrials/2019/trec-pm-2019", ["trec-pm-2019-cds"]),
    Benchmark("trec-covid-rnd5-abstracts", "cord19/trec-covid/round5", ["trec_covid_round5_abstract"]),
    Benchmark("trec-covid-rnd5-fulltexts", "cord19/trec-covid/round5", ["trec_covid_round5_abstract_fulltext"],
              has_fulltexts=True),
    #  Benchmark("trip-click", "", ["trip_click_dataset"], load_from_file=True),
    #  Benchmark("trip-judge", "", ["trip_click_dataset"], load_from_file=True)
]

FIRST_STAGE_TO_PRINT_NAME = {
    "FirstStageGraphRetriever": "Full Match",
    "FirstStagePartialGraphRetriever": "Partial Match"
}

RANKING_STRATEGIES = [
    ConfidenceDocumentRanker(),
    ConfidenceAvgDocumentRanker(),
    ConfidenceMaxDocumentRanker(),
    ConnectivityDocumentRanker(),
    ConnectivityInverseDocumentRanker(),
    ConnectivityNormalizedDocumentRanker(),
    SentenceWeightRanker(),
    DocLengthDocumentRanker(),
    EqualDocumentRanker(),
    RelationalSimDocumentRanker(),
    RelationalSimTFIDFDocumentRanker(),
    RelationalSimTranslationDocumentRanker(),
    RelationalSimNormalizedDocumentRanker(),
    TfIdfAvgDocumentRanker(),
    TfIdfMinDocumentRanker(),
    TfIdfMaxDocumentRanker(),
    TfIdfOnlyConceptsMaxDocumentRanker(),
    TfIdfPlusConceptsMaxDocumentRanker(),
    TranslationDocumentRanker(),
    TfAvgDocumentRanker(),
    TfSumDocumentRanker(),
    TfMinDocumentRanker(),
    TfMaxDocumentRanker(),
    IDFAvgDocumentRanker(),
    IDFSumDocumentRanker(),
    IDFMaxDocumentRanker(),
    IDFMinDocumentRanker(),
    ConceptPositionDocumentRanker(),
    ConceptCoverageDocumentRanker(),
    BM25Text()
]

CONCEPT_STRATEGIES = [
    #  "hybrid",
    #  "expanded",
    #  "exact",
    "likesimilarity",
    "likesimilarityontology"
]

CONCEPT_STRATEGY_2_REAL_NAME = {
    "likesimilarity": "LIKE",
    "likesimilarityontology": "LIKE + Ontology"

}

FIRST_STAGES = [
    FirstStageGraphRetriever(),
    FirstStagePartialGraphRetriever(),
    # FirstStageGraphRetrieverPM2020DrugInPattern(),
    #   FirstStageGraphRetrieverPM2020DrugInDoc(),
    #  FirstStageGraphRetrieverPM2020()
    #   FirstStageKeyword2GraphRetriever(),
    #   FirstStageKeyword2PartialGraphRetriever()
]

FIRST_STAGE_NAMES = [fs.name for fs in FIRST_STAGES]

RANKER_BASE_AVG = [
    "ConfidenceDocumentRanker",
    "TfIdfAvgDocumentRanker",
    "RelationalSimDocumentRanker",
    "ConceptCoverageDocumentRanker"
]
RANKER_BASE_MIN = [
    "ConfidenceDocumentRanker",
    "TfIdfMinDocumentRanker",
    "RelationalSimDocumentRanker",
    "ConceptCoverageDocumentRanker"
]
RANKER_BASE_MAX = [
    "ConfidenceDocumentRanker",
    "TfIdfMaxDocumentRanker",
    "RelationalSimDocumentRanker",
    "ConceptCoverageDocumentRanker"
]

WEIGHT_MATRIX = [
    [1.0 / len(RANKER_BASE_MIN)] * len(RANKER_BASE_MIN)
]

RANKER_BASES = [RANKER_BASE_MIN, RANKER_BASE_AVG, RANKER_BASE_MAX]

SKIPPED_TOPICS = {
    "trec-pm-2017-abstracts": [],
    "trec-pm-2018-abstracts": [],
    "trec-pm-2019-abstracts": [],
    "trec-pm-2020-abstracts": [],
    "trec-pm-2017-cds": [],
    "trec-pm-2018-cds": [],
    "trec-pm-2019-cds": [],
    "trec-covid-rnd5-abstracts": [],
    "trec-covid-rnd5-fulltexts": [],
    "trip-click": [],
    "trip-judge": []
}
IGNORE_DEMOGRAPHIC = False
USE_FRAGMENT_TRANSLATION_SCORE = True
FIRST_STAGE_CUTOFF = 0
MINIMUM_COMPONENTS_IN_QUERY = 2
EVALUATION_SKIP_BAD_TOPICS = True
JUDGED_DOCS_ONLY_FLAG = True

print('==' * 60)
print(f'Ignore demographic                     : {IGNORE_DEMOGRAPHIC}')
print(f'Use document fragment translation score: {USE_FRAGMENT_TRANSLATION_SCORE}')
print(f'First stage cutoff at                  : {FIRST_STAGE_CUTOFF}')
print(f'Minimum required components in query   : {MINIMUM_COMPONENTS_IN_QUERY}')
print(f'Minimum query translation threshold    : {MINIMUM_TRANSLATION_THRESHOLD}')
print(f'Skip bad topics (translation and comp.): {EVALUATION_SKIP_BAD_TOPICS}')
print(f'Judged documents only flag             : {JUDGED_DOCS_ONLY_FLAG}')
print(f'Used concept translation similarity    : {CONCEPT_TRANSLATION_SIMILARITY}')

print('==' * 60)
