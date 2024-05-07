import logging
from typing import List

from narrant.entity.meshontology import MeSHOntology
from narrant.entitylinking.enttypes import DISEASE, HEALTH_STATUS
from narraplay.documentranking.entity_tagger_like import EntityTaggerLike, ScoredEntity

EXPANSION_STEPS = 3


class EntityTaggerLikeOnt:
    __instance = None

    @staticmethod
    def instance():
        if EntityTaggerLikeOnt.__instance is None:
            EntityTaggerLikeOnt()
        return EntityTaggerLikeOnt.__instance

    def __init__(self):
        if EntityTaggerLikeOnt.__instance is not None:
            raise Exception('This class is a singleton - use EntityTaggerLikeOnt.instance()')
        else:
            self.expansion_steps = EXPANSION_STEPS
            self.mesh_ontology = MeSHOntology()
            self.entity_tagger_like = EntityTaggerLike.instance()
            self.min_sim_concept_translation_threshold = 0
            EntityTaggerLikeOnt.__instance = self

    def set_min_sim_concept_translation_threshold(self, min_sim_translation_threshold: float):
        self.min_sim_concept_translation_threshold = min_sim_translation_threshold
        self.entity_tagger_like.set_min_sim_concept_translation_threshold(min_sim_translation_threshold)

    def expand_entities_by_ontology(self, entities: List[ScoredEntity], root_node: str, entity_type: str) \
            -> List[ScoredEntity]:
        expanded_entities = entities.copy()
        for entity in entities:
            # only mesh entities can be expanded
            if entity.entity_type == entity_type and entity.entity_id.startswith('MESH:D'):
                # remove mesh part here
                descriptor_id = entity.entity_id[5:]

                try:
                    for tree_no in self.mesh_ontology.get_tree_numbers_for_descriptor(descriptor_id):
                        if tree_no.startswith(root_node):
                            for step in range(1, tree_no.count('.') + 1):
                                # mesh tree numbers are structured like
                                # C04.213.123.123 (remove the last part).
                                # this is our new super descriptor
                                super_tree_no = '.'.join([t for t in tree_no.split('.')[:-step]])
                                super_descriptor = self.mesh_ontology.get_descriptor_for_tree_no(super_tree_no)

                                # print(f'Expand to tree number: {super_tree_no}')

                                # we did one ontological step
                                score = (1 / (step + 1)) * entity.score

                                # descriptor [0] -> id / [1] -> heading
                                expanded_entities.append(ScoredEntity(entity_id='MESH:' + super_descriptor[0],
                                                                      entity_type=entity.entity_type,
                                                                      entity_class=entity.entity_class,
                                                                      score=score,
                                                                      synonym=entity.synonym))

                                # add also all sub descriptors of the new super descriptor again
                                for subdescriptor in self.mesh_ontology.find_descriptors_start_with_tree_no(
                                        super_tree_no):
                                    # print(f'Expand tree number {super_tree_no} to subdescriptor {subdescriptor}')
                                    # take the same score as previously
                                    # descriptor [0] -> id / [1] -> heading
                                    expanded_entities.append(ScoredEntity(entity_id='MESH:' + subdescriptor[0],
                                                                          entity_type=entity.entity_type,
                                                                          entity_class=entity.entity_class,
                                                                          score=score,
                                                                          synonym=entity.synonym))


                except KeyError:
                    # some entities might not be arranged in our mesh tree (old descriptors for instance)
                    pass

        return [e for e in expanded_entities if e.score > self.min_sim_concept_translation_threshold]

    def expand_cancer_entities_by_ontology(self, entities: List[ScoredEntity]) -> List[ScoredEntity]:
        return self.expand_entities_by_ontology(entities, "C04.", DISEASE)

    def expand_patient_entities_by_ontology(self, entities: List[ScoredEntity]) -> List[ScoredEntity]:
        return self.expand_entities_by_ontology(entities, "M01.", HEALTH_STATUS)

    def tag_entity(self, term: str) -> List[ScoredEntity]:
        entities = self.entity_tagger_like.tag_entity(term)
        if len(entities) == 0:
            return entities
        # expand entities by each step
        #  for step in range(1, self.expansion_steps + 1):
        # print(f'Step: {step}')
        # expand by cancer ontology
        entities = self.expand_cancer_entities_by_ontology(entities)
        # expand patient ontology
    #    entities = self.expand_patient_entities_by_ontology(entities)

        # select the best scored match for each entity id (this will shorten the previous list)
        entities = self.entity_tagger_like.select_best_entity_ids_by_score(entities)

        return entities


def main():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)
    entity_tagger = EntityTaggerLikeOnt.instance()

    tests = ['melanoma']

    for test in tests:
        print()
        for e in entity_tagger.tag_entity(test):
            print(e)


if __name__ == "__main__":
    main()
