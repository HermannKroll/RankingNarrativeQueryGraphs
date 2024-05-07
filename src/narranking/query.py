from itertools import product, combinations
from typing import Set, List

import networkx
import nltk

from narraint.frontend.entity.entitytagger import EntityTagger
from narrant.entity.entity import Entity
from narraplay.documentranking.benchmark import Topic
from narraplay.documentranking.document import AnalyzedNarrativeDocument
from narraplay.documentranking.entity_tagger_like import EntityTaggerLike
from narraplay.documentranking.entity_tagger_like_ont import EntityTaggerLikeOnt

stopwords = set(nltk.corpus.stopwords.words('english'))
trans_map = {p: ' ' for p in '[]()?!'}  # PUNCTUATION}
translator = str.maketrans(trans_map)


class AnalyzedQuery:

    def __init__(self, topic: Topic, concept_strategy, translate_query=True):
        self.topic = topic

        if not translate_query:
            return

        self.concept_strategy = concept_strategy
        self.concepts = set()
        self.partial_concepts = list()
        self.concept2score = dict()
        if concept_strategy in ["exact", "expanded", "hybrid"]:
            self.tagger = EntityTagger()
        elif concept_strategy in ["likesimilarity"]:
            self.taggerV2 = EntityTaggerLike.instance()
        elif concept_strategy in ["likesimilarityontology"]:
            self.taggerV2 = EntityTaggerLikeOnt.instance()
        else:
            raise ValueError(f"{concept_strategy} concept strategy is not supported")
        self.component2concepts = dict()
        self.component2concepts_with_type = dict()
        for and_component in self.topic.get_query_components():
            concepts_for_component = set()
            concepts_with_type_for_component = set()
            comp_names = []
            for or_component, or_concept_type_constraints in and_component:
                comp_names.append(or_component)

                if concept_strategy in ["exact", "expanded", "hybrid"]:
                    concepts = self.__greedy_find_concepts_in_keywords_v1(or_component)
                elif concept_strategy in ["likesimilarity", "likesimilarityontology"]:
                    concepts = self.__greedy_find_concepts_in_keywords_v2(or_component)
                    for c in concepts:
                        if c.entity_id not in self.concept2score:
                            self.concept2score[c.entity_id] = c.score
                        else:
                            # concept is only as good as the best found translation
                            self.concept2score[c.entity_id] = max(self.concept2score[c.entity_id], c.score)
                else:
                    raise ValueError(f"{concept_strategy} concept strategy is not supported")

                if or_concept_type_constraints:
                    # Filter by constraints and only keep ids
                    concepts_with_type = set([c for c in concepts if c.entity_type in or_concept_type_constraints])
                    concepts = set([c.entity_id for c in concepts if c.entity_type in or_concept_type_constraints])
                else:
                    concepts_with_type = set(concepts)
                    concepts = set([c.entity_id for c in concepts])
                self.concepts.update(concepts)
                concepts_for_component.update(concepts)
                concepts_with_type_for_component.update(concepts_with_type)

            self.partial_concepts.append(concepts_for_component)
            component_str = ' || '.join(sorted([c for c in comp_names]))
            self.component2concepts[component_str] = list(concepts_for_component)
            self.component2concepts_with_type[component_str] = list(concepts_with_type_for_component)

        if len(self.partial_concepts) > 0:
            self.partial_weight = 1 / len(self.partial_concepts)
        else:
            self.partial_weight = 0.0

        # all concepts have the same score
        if concept_strategy in ["exact", "expanded", "hybrid"]:
            for c in self.concepts:
                self.concept2score[c] = 1.0

    def get_query_translation_score(self):
        # we did not translate all components
        if len(list(self.topic.get_query_components())) > len(self.component2concepts):
            return 0.0
        if self.concept_strategy in ["likesimilarity", "likesimilarityontology"]:
            # find the best translation for each component
            max_trans_scores = []
            for component, concepts in self.component2concepts_with_type.items():
                if concepts:
                    max_trans_scores.append(max([e.score for e in concepts]))
                else:
                    max_trans_scores.append(0.0)

            # query is as good as its worst translation
            return min(max_trans_scores)
        else:
            return 1.0

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

    def __greedy_find_concepts_in_keywords_v2(self, concept_name):
        try:
            return self.taggerV2.tag_entity(concept_name)
        except KeyError:
            return []

    def __find_entities_in_string(self, concept_name):
        if self.concept_strategy == "exact":
            return self.tagger.tag_entity(concept_name, expand_search_by_prefix=False)
        elif self.concept_strategy == "expanded":
            return self.tagger.tag_entity(concept_name, expand_search_by_prefix=True)
        elif self.concept_strategy == "hybrid":
            try:
                return self.tagger.tag_entity(concept_name, expand_search_by_prefix=False)
            except KeyError:
                return self.tagger.tag_entity(concept_name, expand_search_by_prefix=True)

    def __greedy_find_concepts_in_keywords_v1(self, keywords) -> Set[Entity]:
        try:
            return self.__find_entities_in_string(keywords)
        except KeyError:
            return set()

    def __greedy_find_concepts_in_keywords_with_search(self, keywords) -> Set[Entity]:
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
