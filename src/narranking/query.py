from itertools import product, combinations
from typing import Set, List

import networkx
import nltk

from narraint.frontend.entity.entitytagger import EntityTagger
from narrant.entity.entity import Entity
from narranking.benchmark import Topic
from narranking.document import AnalyzedNarrativeDocument

stopwords = set(nltk.corpus.stopwords.words('english'))
trans_map = {p: ' ' for p in '[]()?!'}  # PUNCTUATION}
translator = str.maketrans(trans_map)


class AnalyzedQuery:

    def __init__(self, topic: Topic, concept_strategy):
        self.topic = topic
        self.concept_strategy = concept_strategy
        self.concepts = set()
        self.partial_concepts = list()
        self.tagger = EntityTagger.instance()
        self.component2concepts = dict()
        for and_component in self.topic.get_query_components():
            concepts_for_component = set()
            comp_names = []
            for or_component, or_concept_type_constraints in and_component:
                comp_names.append(or_component)
                concepts = self.__greedy_find_concepts_in_keywords(or_component)
                if or_concept_type_constraints:
                    # Filter by constraints and only keep ids
                    concepts = set([c.entity_id for c in concepts if c.entity_type in or_concept_type_constraints])
                else:
                    concepts = set([c.entity_id for c in concepts])
                self.concepts.update(concepts)
                concepts_for_component.update(concepts)

            self.partial_concepts.append(concepts_for_component)
            component_str = ' || '.join(sorted([c for c in comp_names]))
            self.component2concepts[component_str] = list(concepts_for_component)
        self.partial_weight = 1 / len(self.partial_concepts)

    def get_statistics(self):
        return dict(topic=str(self.topic), component2concepts=self.component2concepts)

    def get_document_statistics(self, documents: List[AnalyzedNarrativeDocument]):
        documents_per_concept = {c: 0 for c in self.component2concepts.keys()}
        documents_per_subj_obj = {c: 0 for c in self.component2concepts.keys()}
        documents_per_statement = {c: 0 for c in self.component2concepts.keys()}
        component_combinations = list(combinations(self.component2concepts.keys(), 2))
        statements_per_document = {s: 0 for s in range(len(component_combinations) + 1)}
        query_comps_in_documents = {s: 0 for s in range(len(self.component2concepts) + 1)}
        connected_query_comps_in_documents = {s: 0 for s in range(len(component_combinations) + 1)}
        # concepts in documents
        for d in documents:
            for component, concepts in self.component2concepts.items():
                if component in d.concept2frequency:
                    documents_per_concept[component] += 1
                    continue
                else:
                    for concept in concepts:
                        if concept in d.concept2frequency:
                            documents_per_concept[component] += 1
                            break
            for component, concepts in self.component2concepts.items():
                if component in d.subjects or component in d.objects:
                    documents_per_subj_obj[component] += 1
                    continue
                else:
                    for concept in concepts:
                        if concept in d.subjects or concept in d.objects:
                            documents_per_subj_obj[component] += 1
                            break
            statement_combinations = 0
            for cp_subj, cp_obj in component_combinations:
                # check if any combination of the partial concepts is a known statement
                for subj, obj in list(product(self.component2concepts[cp_subj], self.component2concepts[cp_obj])):
                    if (subj, obj) in d.so2statement:
                        documents_per_statement[cp_subj] += 1
                        documents_per_statement[cp_obj] += 1
                        statement_combinations += 1
                        break
            statements_per_document[statement_combinations] += 1

            # Compute how many components are connected on the graph structure
            graph: networkx.MultiGraph = networkx.MultiGraph()
            for stmt in d.extracted_statements:
                graph.add_edge(stmt.subject_id, stmt.object_id)

            connected_components = 0
            for cp_subj, cp_obj in component_combinations:
                # check if any combination of the partial concepts is a known statement
                found = False
                # check if one of the possible subjects and objects is connected
                for subj, obj in list(product(self.component2concepts[cp_subj], self.component2concepts[cp_obj])):
                    # skip if one of the values does not exist as a node
                    if subj not in graph or obj not in graph:
                        continue
                    if networkx.has_path(graph, subj, obj):
                        found = True
                        break
                if found:
                    connected_components += 1

            connected_query_comps_in_documents[connected_components] += 1

            query_parts_in_doc = 0
            for component, concepts in self.component2concepts.items():
                if len(set(concepts).intersection(d.nodes)) > 0:
                    query_parts_in_doc += 1
            query_comps_in_documents[query_parts_in_doc] += 1

        return dict(documents=len(documents), documents_per_concept=documents_per_concept,
                    documents_per_subj_obj=documents_per_subj_obj, documents_per_statement=documents_per_statement,
                    statements_per_document=statements_per_document, query_comps_in_documents=query_comps_in_documents,
                    connected_query_comps_in_documents=connected_query_comps_in_documents)

    def __find_entities_in_string(self, concept_name):
        if self.concept_strategy == "exac":
            return self.tagger.tag_entity(concept_name, expand_search_by_prefix=False)
        elif self.concept_strategy == "expc":
            return self.tagger.tag_entity(concept_name, expand_search_by_prefix=True)
        elif self.concept_strategy == "hybrid":
            try:
                return self.tagger.tag_entity(concept_name, expand_search_by_prefix=False)
            except KeyError:
                return self.tagger.tag_entity(concept_name, expand_search_by_prefix=True)

    def __greedy_find_concepts_in_keywords(self, keywords) -> Set[Entity]:
        resulting_concepts = set()
        try:
            entities_in_part = self.__find_entities_in_string(keywords)
            resulting_concepts.update({e for e in entities_in_part})
        except KeyError:
            # perform a backward search till something was found
            keywords = keywords.strip().split(' ')
            if len(keywords) > 0:
                found = False
                for j in range(len(keywords) - 1, 0, -1):
                    if found:
                        break
                    current_part = ' '.join([k for k in keywords[j:]])
                    try:
                        entities_in_part = self.__find_entities_in_string(current_part)
                        resulting_concepts.update({e for e in entities_in_part})
                        # print(f'Check part: {current_part} -> found: {entities_in_part}')
                        found = True
                    except KeyError:
                        pass

                # Perform a forward search till something was found
                found = False
                for j in range(1, len(keywords) + 1, 1):
                    if found:
                        break
                    current_part = ' '.join([k for k in keywords[:j]])
                    try:
                        entities_in_part = self.__find_entities_in_string(current_part)
                        resulting_concepts.update({e for e in entities_in_part})
                        #  print(f'Check part: {current_part} -> found: {entities_in_part}')
                        found = True
                        # self.concepts.update({e.entity_type for e in entities_in_part})
                    except KeyError:
                        pass

        return resulting_concepts

    def to_dict(self):
        return {"topic": str(self.topic),
                "concepts": str(self.concepts)}
