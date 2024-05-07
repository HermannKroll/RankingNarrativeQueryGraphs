import json
import os.path
import re
from abc import abstractmethod, ABC
from typing import List
from xml.etree import ElementTree

import ir_datasets
import pytrec_eval as pe
import tqdm
from nltk.corpus import stopwords

from narrant.entitylinking.enttypes import SPECIES
from narraplay.documentranking.config import DATA_DIR, PM2020_TOPIC_FILE, PUBMED_BASELINE_ID_DIR, \
    TRIP_CLICK_TOPIC_FILES, TRIP_JUDGE_TOPIC_FILES, RUNS_DIR, QRELS_PATH, DATASET_TO_PUBMED_BASE_ID_FILE, \
    RESULT_DIR_TOPICS, MINIMUM_TRANSLATION_THRESHOLD
from narraplay.documentranking.entity_tagger_like import EntityTaggerLike

DRUG = "Drug"
CHEMICAL = "Chemical"
DISEASE = "Disease"
HEALTH_STATUS = "HealthStatus"
TARGET = "Target"
GENE = "Gene"

STOPWORDS = set(stopwords.words('english'))
# single characters are still important (e.g. D for Vitamin D)
STOPWORDS = {s for s in STOPWORDS if len(s) > 1}


class Topic(ABC):
    @abstractmethod
    def __init__(self, query):
        self.query_id = query.query_id

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_query_components(self) -> [[str]]:
        """
        :return: keywords joined as string, separated by a space
        """
        pass

    @abstractmethod
    def get_benchmark_string(self) -> str:
        raise NotImplementedError

    def get_drug_name(self):
        raise NotImplementedError


def demographic_rule(demographic: str) -> str:
    return "Patients"
    # We don't have more specific concepts here


def demographic_rule_v2(demographic: str) -> str:
    group = demographic.split(' ')[-1]
    if group == "male":
        return "men"
    if group == "female":
        return "women"
    return group
    # We don't have more specific concepts here


def gene_rule(gene: str):
    # only take upper letters
    # some genes and mutations are not divided by a space, e.g. 'AKT1(E17K)'
    gene_i = gene.replace('(', ' ')
    gene_i = ''.join([c for c in gene_i if c.isupper() or c.isdigit() or c in [' ', ',', '(', '-', ')']])
    # split by ',' if contained + split by ' ' and take only the first part
    # e.g. BRAF (V600E) is reduced to BRAF because we do not support specific mutations
    if ',' in gene_i:
        yield from [[(gene.strip().split(' ')[0].strip(), [TARGET, GENE]) for gene in gene_i.split(',')]]
    # divide topics SND1-BRAF fusion  to SND1 and BRAF
    elif '-' in gene_i and 'fusion' in gene.lower():
        yield from [[(gene.strip().split(' ')[0].strip(), [TARGET, GENE]) for gene in gene_i.split('-')]]
    # then just take the main component
    else:
        yield from [[(gene_i.strip().split(' ')[0].strip(), [TARGET, GENE])]]


def gene_rule_text(gene):
    gene_comps = list(gene_rule(gene))
    if len(gene_comps) > 0:
        gene_comps = gene_comps[0]
        return ' '.join([c[0] for c in gene_comps]).strip()
    else:
        return ''


def greedy_search_find_query_components(query_text: str, tagger_type=EntityTaggerLike):
    tagger = tagger_type.instance()
    query_text = query_text.lower()

    title_parts = [p.strip() for p in query_text.split() if p.strip() != "" and p not in STOPWORDS]
    title_sub_parts = title_parts
    components = []

    while True:
        # create new string to tag
        entity_string = " ".join(title_sub_parts)

        # try to tag entities in that string
        try:
            entities = tagger.tag_entity(entity_string)
        except KeyError:
            entities = []

        if len(entities) > 0 and max(e.score for e in entities) >= MINIMUM_TRANSLATION_THRESHOLD:
            # the tagger found a translation for the entity string
            components.append([(entity_string, None)])

            # remove the first len(title_sub_parts) elements of the title to tag the remaining part
            title_parts = title_parts[len(title_sub_parts):]
            title_sub_parts = title_parts

        elif len(title_sub_parts) == 1:
            # the first element of the title cant be tagged
            title_parts = title_parts[1:]
            title_sub_parts = title_parts
        else:
            # decrease the size of the tagged string
            title_sub_parts = title_sub_parts[:-1]

        if len(title_parts) == 0:
            break
    return components


class PrecMed2017Topic(Topic):
    def __init__(self, query):
        super().__init__(query)
        self.disease = query.disease
        self.gene = query.gene
        self.demographic = query.demographic
        self.other = query.other
        if self.other == "None":
            self.other = None

    def __str__(self):
        return f'<{self.query_id} disease={self.disease} gene={self.gene} demographic={self.demographic} other={self.other}>'

    def get_query_components(self) -> [str]:
        yield [(self.disease, [DISEASE])]
        yield from gene_rule(self.gene)
        yield [(demographic_rule_v2(self.demographic), [HEALTH_STATUS, SPECIES])]
        # if self.other:
        #    yield from [[(s.strip(), [DISEASE]) for s in self.other.split(',')]]

    def get_benchmark_string(self):
    #    if self.other:
    #        return f'{self.disease} {self.gene} {self.demographic} {self.other}'
     #   else:
        return f'{self.disease} {self.gene} {self.demographic}'


class PrecMed2018Topic(Topic):
    def __init__(self, query):
        super().__init__(query)
        self.disease = query.disease
        self.gene = query.gene
        self.demographic = query.demographic

    def __str__(self):
        return f'<{self.query_id} disease={self.disease} gene={self.gene} demographic={self.demographic}>'

    def get_query_components(self) -> [str]:
        yield [(self.disease, [DISEASE])]
        yield from gene_rule(self.gene)
        yield [(demographic_rule_v2(self.demographic), [HEALTH_STATUS, SPECIES])]

    def get_benchmark_string(self):
        return f'{self.disease} {self.gene} {self.demographic}'


class TrecCOVIDTopic(Topic):
    def __init__(self, query):
        super().__init__(query)
        self.title = query.title
        self.narrative = query.narrative
        self.description = query.description

    def __str__(self):
        return f'<{self.query_id} title={self.title}>'

    def get_query_components(self) -> [str]:
        yield from greedy_search_find_query_components(self.title)

    def get_benchmark_string(self):
        return self.title


class PrecMed2020Topic(Topic):

    def __init__(self, number, disease, gene, treatment):
        self.query_id = number
        self.disease = disease
        self.gene = gene
        self.treatment = treatment

    def __str__(self):
        return f'<{self.query_id} disease={self.disease} gene={self.gene} treatment={self.treatment}>'

    def rename_cancer(self, disease):
        if disease == "non-small cell carcinoma":
            return "non-small cell lung cancer"
        elif disease == "ovarian carcinoma":
            return "ovarian cancer"
        else:
            return disease

    def get_query_components(self) -> [str]:
        yield from [[(self.rename_cancer(self.disease), [DISEASE])], [(self.gene, [GENE, TARGET])],
                    [(self.treatment, [DRUG, CHEMICAL])]]

    def get_benchmark_string(self):
        return f'{self.disease} {self.gene} {self.treatment}'

    @staticmethod
    def parse_topics(path_to_file=PM2020_TOPIC_FILE):
        with open(path_to_file) as file:
            root = ElementTree.parse(file).getroot()

        if root is None:
            return

        topics = []
        for topic in root.findall('topic'):
            number = topic.get('number')
            disease = topic.find('disease').text.strip()
            gene = topic.find('gene').text.strip()
            treatment = topic.find('treatment').text.strip()
            topics.append(PrecMed2020Topic(number, disease, gene, treatment))
        return topics

    def get_drug_name(self):
        return self.treatment


class TripClickTopic(Topic):
    def __init__(self, query_id, title, components):
        self.query_id = query_id
        self.title = title
        self.components = components

    def __str__(self):
        return f'<{self.query_id} title={self.title} components={len(self.components)}>'

    def get_query_components(self) -> [str]:
        yield from self.components

    def get_benchmark_string(self) -> str:
        return self.title

    @staticmethod
    def parse_topics(path_to_files=TRIP_CLICK_TOPIC_FILES):
        return TripClickTopic.try_load_from_cache(path_to_files, "trip-click", TripClickTopic)

    @staticmethod
    def try_load_from_cache(path_to_files, benchmark, topic_type):
        cache_file_path = os.path.join(RESULT_DIR_TOPICS, f"{benchmark}.json")
        topics = list()
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "rt") as file:
                topic_dump = json.load(file)

            for query_id, topic in topic_dump.items():
                topics.append(topic_type(query_id, topic["title"], topic["components"]))
        else:
            print(f"{benchmark.title()} topics have to be parsed first. It may take a while...")
            top_pattern = re.compile(r"(?s)<top>\n*(.*?)\n*</top>")

            for i, topics_file in enumerate(path_to_files):
                with open(topics_file, "rt") as file:
                    topics_raw = re.findall(top_pattern, file.read())
                    for topic_str in tqdm.tqdm(topics_raw, desc=f"file {i + 1}/{len(path_to_files)}"):
                        query_id, topic_title, *_ = [s for s in topic_str.split('\n') if len(s) > 0]
                        query_id = query_id.split(':')[-1].strip()
                        topic_title = topic_title.split('> ')[-1].strip()
                        components = greedy_search_find_query_components(topic_title)
                        topics.append(topic_type(int(query_id), topic_title, components))

            print(f"Finished parsing topics. Saving into {cache_file_path}")
            with open(cache_file_path, "wt") as file:
                obj = {t.query_id: {"title": t.title, "components": t.components} for t in topics}
                json.dump(obj, file, indent=2)
        return topics


class TripJudgeTopic(TripClickTopic):
    @staticmethod
    def parse_topics(path_to_files=TRIP_JUDGE_TOPIC_FILES):
        return TripJudgeTopic.try_load_from_cache(path_to_files, "trip-judge", TripJudgeTopic)


DATASET_TO_TOPIC = {
    "medline/2017/trec-pm-2017": PrecMed2017Topic,
    "medline/2017/trec-pm-2018": PrecMed2018Topic,
    "clinicaltrials/2017/trec-pm-2017": PrecMed2017Topic,
    "clinicaltrials/2017/trec-pm-2018": PrecMed2018Topic,
    "clinicaltrials/2019/trec-pm-2019": PrecMed2018Topic,
    "cord19/trec-covid/round5": TrecCOVIDTopic
}

DATASET_TO_FILE_LOADING = {
    "trec-pm-2020-abstracts": PrecMed2020Topic.parse_topics,
    "trip-click": TripClickTopic.parse_topics,
    "trip-judge": TripJudgeTopic.parse_topics
}


class Benchmark:

    def __init__(self, name, ir_dataset_name, document_collections: List[str], load_from_file=False,
                 has_fulltexts=False):
        self.name = name
        self.ir_dataset_name = ir_dataset_name
        self.document_collections = document_collections
        self.topics: [Topic] = []
        self.topic_id2docs = {}
        self.has_fulltexts = has_fulltexts

        self.documents_for_baseline = None
        self.documents_for_baseline_load = False
        if load_from_file:
            self.topics = DATASET_TO_FILE_LOADING[name]()

        else:
            dataset = ir_datasets.load(ir_dataset_name)
            for query in dataset.queries_iter():
                self.topics.append(DATASET_TO_TOPIC[ir_dataset_name](query))

        json_path = os.path.join(DATA_DIR, "json")
        json_path = os.path.join(json_path, f'{name}.json')
        with open(json_path, 'rt') as f:
            json_data = json.load(f)
            for topic, documents in json_data.items():
                self.topic_id2docs[topic] = documents

            # print(f'{topic} has {len(documents)} documents to rank')
        self.filter_benchmark_for_qrel_topics()

    def filter_benchmark_for_qrel_topics(self):
        path = os.path.join(RUNS_DIR, QRELS_PATH[self.name])
        print(f'Loading qrels from {path}')
        with open(path, 'r') as file:
            qrels = pe.parse_qrel(file)

        no_before = len(self.topics)
        self.topics = [t for t in self.topics if str(t.query_id) in qrels]
        print(f'Applied qrel-filtering of topics ({no_before} -> {len(self.topics)})')

    def get_documents_for_baseline(self):
        if not self.documents_for_baseline_load:
            if self.name in DATASET_TO_PUBMED_BASE_ID_FILE:
                path = os.path.join(PUBMED_BASELINE_ID_DIR, DATASET_TO_PUBMED_BASE_ID_FILE[self.name])
                self.documents_for_baseline = set()
                with open(path, 'rt') as f:
                    for line in f:
                        self.documents_for_baseline.add(int(line.strip()))
                print(f'Load {len(self.documents_for_baseline)} for {self.name}')
            self.documents_for_baseline_load = True
        return self.documents_for_baseline

    def get_relevant_documents(self):
        relevant_documents = set()
        for _, doc_ids in self.topic_id2docs.items():
            relevant_documents = relevant_documents.union(doc_ids)
        return relevant_documents

    def __str__(self):
        return f'<{self.name} with dataset {self.ir_dataset_name}>'

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    # call TripClick benchmark to create component cache topics
    b = Benchmark("trip-click", "", ["PubMed"], load_from_file=True)
    print(b.name, len(b.topics))

    # call TripJudge benchmark to create component cache topics
    b = Benchmark("trip-judge", "", ["PubMed"], load_from_file=True)
    print(b.name, len(b.topics))
