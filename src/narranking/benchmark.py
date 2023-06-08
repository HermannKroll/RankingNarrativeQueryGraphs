import json
import os.path
from abc import abstractmethod, ABC
from typing import List
from xml.etree import ElementTree

import ir_datasets

from narranking.config import DATA_DIR, PM2020_TOPIC_FILE, PUBMED_BASELINE_ID_DIR, IGNORE_DEMOGRAPHIC


DRUG = "Drug"
CHEMICAL = "Chemical"
DISEASE = "Disease"
HEALTH_STATUS = "HealthStatus"
TARGET = "Target"
GENE = "Gene"


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
    def get_benchmark_string(self, gene_modified, demographic) -> str:
        raise NotImplementedError


def demographic_rule(demographic: str) -> str:
    return "Patients"
    # We don't have more specific concepts here


def gene_rule(gene: str):
    # only take upper letters
    gene_i = gene.replace('(', ' ')
    gene_i = ''.join([c for c in gene_i if c.isupper() or c.isdigit() or c in [' ', ',', '(', '-']])
    if gene_i.strip() and len([c for c in gene_i if c.isupper()]) > 0:
        if ',' in gene_i:
            yield from [[(gene.strip().split(' ')[0].strip(), [TARGET, GENE]) for gene in gene_i.split(',')
                         if gene.strip().split(' ')[0].strip()]]
        elif '-' in gene_i:
            yield from [[(gene.strip().split(' ')[0].strip(), [TARGET, GENE]) for gene in gene_i.split('-')
                         if gene.strip().split(' ')[0].strip()]]
        else:
            gene = gene_i.strip().split(' ')[0].strip()
            yield [(gene, [TARGET, GENE])]


def gene_rule_text(gene):
    gene_comps = list(gene_rule(gene))
    if len(gene_comps) > 0:
        gene_comps = gene_comps[0]
        return ' '.join([c[0] for c in gene_comps]).strip()
    else:
        return ''


#  demographic = demographic.split(' ')[1].strip()
# if demographic == "female":
#    demographic = "women"
# elif demographic == "male":
#   demographic = "men"
# return demographic


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
        if not IGNORE_DEMOGRAPHIC:
            demographic = demographic_rule(self.demographic)
            yield [(demographic, HEALTH_STATUS)]

        if self.other:
            yield from [[(s.strip(), DISEASE) for s in self.other.split(',')]]

    def get_benchmark_string(self, gene_modified, demographic):
        parts = [self.disease]

        if gene_modified:
            gene_mod = gene_rule_text(self.gene)
            if gene_mod:
                parts.append(gene_mod)
        else:
            parts.append(self.gene)

        if demographic:
            parts.append(self.demographic)

        if self.other:
            parts.append(self.other.replace(',', ' '))

        return ' '.join([p for p in parts])


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
        if not IGNORE_DEMOGRAPHIC:
            demographic = demographic_rule(self.demographic)
            yield [(demographic, HEALTH_STATUS)]

    def get_benchmark_string(self, gene_modified, demographic):
        parts = [self.disease]

        if gene_modified:
            gene_mod = gene_rule_text(self.gene)
            if gene_mod:
                parts.append(gene_mod)
        else:
            parts.append(self.gene)

        if demographic:
            parts.append(self.demographic)

        return ' '.join([p for p in parts])


class TrecCOVIDTopic(Topic):

    def __init__(self, query):
        super().__init__(query)
        self.title = query.title
        self.title = self.title.replace('coronavirus', 'covid-19')
        self.narrative = query.narrative
        self.description = query.description

    def __str__(self):
        return f'<{self.query_id} title={self.title}>'

    def get_query_components(self) -> [str]:
        title_lower: str = self.title.lower()
        title_lower = title_lower.replace(' and ', '')
        if 'covid' in title_lower and "covid-19" not in title_lower:
            title_lower = title_lower.replace('covid', 'covid-19')
        if 'covid-19' in title_lower:
            splits = title_lower.split("covid-19")
            for s in splits:
                s = s.replace('covid-19', '')
                if s.strip():
                    yield [(s.strip(), None)]
            yield [("covid-19", DISEASE)]
        else:
            yield [(self.title, None)]

    def get_benchmark_string(self, gene_modified, demographic):
        return self.title


class PrecMed2020Topic(Topic):

    def __init__(self, number, disease, gene, treatment):
        self.query_id = number
        self.disease = disease
        self.gene = gene
        self.treatment = treatment

    def __str__(self):
        return f'<{self.query_id} disease={self.disease} gene={self.gene} treatment={self.treatment}>'

    def get_query_components(self) -> [str]:
        yield from [[(self.disease, [DISEASE])], [(self.gene, [GENE, TARGET])], [(self.treatment, [DRUG, CHEMICAL])]]

    def get_benchmark_string(self, gene_modified, demographic):
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


DATASET_TO_TOPIC = {
    "medline/2017/trec-pm-2017": PrecMed2017Topic,
    "medline/2017/trec-pm-2018": PrecMed2018Topic,
    "clinicaltrials/2017/trec-pm-2017": PrecMed2017Topic,
    "clinicaltrials/2017/trec-pm-2018": PrecMed2018Topic,
    "clinicaltrials/2019/trec-pm-2019": PrecMed2018Topic,
    "cord19/trec-covid/round5": TrecCOVIDTopic
}

DATASET_TO_FILE_LOADING = {
    "trec-pm-2020-abstracts": PrecMed2020Topic.parse_topics
}

DATASET_TO_PUBMED_BASE_ID_FILE = {
    "trec-pm-2017-abstracts": "pubmed_baseline_pm17.txt",
    "trec-pm-2018-abstracts": "pubmed_baseline_pm17.txt",
    "trec-pm-2019-abstracts": "pubmed_baseline_pm19.txt",
    "trec-pm-2020-abstracts": "pubmed_baseline_pm19.txt"
}


class Benchmark:

    def __init__(self, name, ir_dataset_name, document_collections: List[str], load_from_file=False):
        self.name = name
        self.ir_dataset_name = ir_dataset_name
        self.document_collections = document_collections
        self.topics: [Topic] = []
        self.topic_id2docs = {}

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

    def get_documents_for_baseline(self):
        if self.documents_for_baseline_load:
            return self.documents_for_baseline
        else:
            if self.name in DATASET_TO_PUBMED_BASE_ID_FILE:
                path = os.path.join(PUBMED_BASELINE_ID_DIR, DATASET_TO_PUBMED_BASE_ID_FILE[self.name])
                self.documents_for_baseline = set()
                with open(path, 'rt') as f:
                    for line in f:
                        self.documents_for_baseline.add(int(line.strip()))
                print(f'Load {len(self.documents_for_baseline)} for {self.name}')
            self.documents_for_baseline_load = True

    def __str__(self):
        return f'<{self.name} with dataset {self.ir_dataset_name}>'

    def __repr__(self):
        return self.__str__()

# for dataset_name in DATASET_TO_TOPIC:
#     print(dataset_name)
#     b = Benchmark("trec-pm-2017-abstracts", dataset_name, "PubMed")
#     for q in b.topics:
#         print(q)
#     print('\n\n')
