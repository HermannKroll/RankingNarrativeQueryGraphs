import logging
import string
from typing import List

from Levenshtein import distance as levenshtein_distance

from narraint.backend.database import SessionExtended
from narraint.frontend.entity.entityindexbase import EntityIndexBase
from narrant.entity.entity import Entity
from narraplay.documentranking.models import EntityTaggerData, RANKING_EXTENDED

MINIMUM_CHARACTERS_FOR_TAGGING = 3
CONCEPT_TRANSLATION_SIMILARITY = "jaccard"

class ScoredEntity(Entity):

    def __init__(self, score: float, synonym, entity_id, entity_type, entity_class=None):
        self.score = score
        self.synonym = synonym
        super().__init__(entity_id=entity_id, entity_type=entity_type, entity_class=entity_class)

    def __str__(self):
        return '{} -> {} ({}) with score {}'.format(self.synonym, self.entity_id, self.entity_type, self.score)

    def __repr__(self):
        return self.__str__()


class EntityTaggerLike(EntityIndexBase):
    # Keyword = "melanoma"
    #
    # DB - Table (id, type, synonym)
    #
    # entity_synonym LIKE '%melanoma%' <<<<
    #
    # # BRAF (VE600)
    # LIKE '%BRAF%' AND '%VE600%' < <<<
    # for result in results:
    #     similarity = levensthein(keyword, result.synonym)

    #
    #

    __instance = None

    @staticmethod
    def instance():
        if EntityTaggerLike.__instance is None:
            EntityTaggerLike()
        return EntityTaggerLike.__instance

    def __init__(self):
        if EntityTaggerLike.__instance is not None:
            raise Exception('This class is a singleton - use EntityTagger.instance()')
        else:
            super().__init__()
            trans_map = {p: '' for p in string.punctuation}
            self.__translator = str.maketrans(trans_map)
            logging.info('Initialize entity tagger V2...')
            self.db_values_to_insert = []
            session = SessionExtended.get(declarative_base=RANKING_EXTENDED)
            if 0 == session.query(EntityTaggerData).count():
                self.create_database_table()
            session.remove()
            self.min_sim_concept_translation_threshold = 0.0
            self.string_distance_metric = CONCEPT_TRANSLATION_SIMILARITY
            EntityTaggerLike.__instance = self

    def set_min_sim_concept_translation_threshold(self, min_sim_translation_threshold: float):
        self.min_sim_concept_translation_threshold = min_sim_translation_threshold

    def set_string_distance_measure(self, string_distance_metric: str):
        string_distance_metric = string_distance_metric.lower()
        if string_distance_metric not in {"levenshtein", "jaccard", "jarowinkler"}:
            raise NotImplementedError(f"String distance metric '{string_distance_metric}' is unknown")
        self.string_distance_metric = string_distance_metric

    def prepare_string(self, term: str) -> str:
        return term.strip().lower().translate(self.__translator).strip()

    def create_database_table(self):
        logging.info('Creating database table for EntityTagger index...')
        self._create_index()
        logging.info(f'Inserting {len(self.db_values_to_insert)} values into database...')
        session = SessionExtended.get(declarative_base=RANKING_EXTENDED)
        EntityTaggerData.bulk_insert_values_into_table(session, self.db_values_to_insert)

        self.db_values_to_insert.clear()
        logging.info('Finished')

    def _add_term(self, term, entity_id: str, entity_type: str, entity_class: str = None):
        self.db_values_to_insert.append(dict(entity_id=entity_id,
                                             entity_type=entity_type,
                                             entity_class=entity_class,
                                             synonym=self.prepare_string(term)))

    def select_best_entity_ids_by_score(self, scored_entities: List[ScoredEntity]) -> List[ScoredEntity]:
        # Rank entities by their score
        scored_entities.sort(key=lambda x: x.score, reverse=True)
        # Keep only the best match
        selected_entities = set()
        selected_scored_entities = []
        for entity in scored_entities:
            key = (entity.entity_id, entity.entity_type)
            if key in selected_entities:
                continue

            selected_entities.add(key)
            selected_scored_entities.append(entity)
        return selected_scored_entities

    def tag_entity(self, term: str) -> List[ScoredEntity]:
        # first process the string
        term = self.prepare_string(term)

        # ignore to short terms -> no matches
        if not term or len(term) < MINIMUM_CHARACTERS_FOR_TAGGING:
            raise KeyError('Does not know an entity for term: {}'.format(term))

        session = SessionExtended.get(declarative_base=RANKING_EXTENDED)
        query = session.query(EntityTaggerData)
        # Construct the query as a disjunction with like expressions
        # e.g. the search covid 19 is performed by
        # WHERE synonym like '%covid%' AND synonym like '%19%'
        # SQL alchemy overloads the bitwise & operation to connect different expressions via AND
        filter_exp = None
        for part in term.split(' '):
            part = part.strip()
            if not part:
                continue

            if filter_exp is None:
                filter_exp = EntityTaggerData.synonym.like('%{}%'.format(part))
            else:
                filter_exp = filter_exp & EntityTaggerData.synonym.like('%{}%'.format(part))
        query = query.filter(filter_exp)

        scored_entities = []
        for result in query:
            if self.string_distance_metric == "levenshtein":
                max_string_len = max(len(term), len(result.synonym))
                lev_distance = levenshtein_distance(term, result.synonym)
                similarity = (max_string_len - lev_distance) / max_string_len
            elif self.string_distance_metric == "jaccard":
                term_tokens = set(term.strip().split(' '))
                synonym_tokens = set(result.synonym.strip().split(' '))
                similarity = len(term_tokens.intersection(synonym_tokens)) / len(term_tokens.union(synonym_tokens))
            elif self.string_distance_metric == "jarowinkler":
                raise NotImplementedError("Jarowinkler distance is not implemented yet")
            else:
                raise NotImplementedError(f"String distance metric '{self.string_distance_metric}' is unknown")

            # Default set to 0.0
            if similarity > self.min_sim_concept_translation_threshold:
                scored_entities.append(ScoredEntity(score=similarity,
                                                    synonym=result.synonym,
                                                    entity_id=result.entity_id,
                                                    entity_type=result.entity_type,
                                                    entity_class=result.entity_class))

        session.remove()

        # select the best scored entities
        selected_scored_entities = self.select_best_entity_ids_by_score(scored_entities)
        if len(selected_scored_entities) == 0:
            raise KeyError('Does not know an entity for term: {}'.format(term))
        return selected_scored_entities


def main():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)
    entity_tagger = EntityTaggerLike.instance()

    tests = ['covid', 'covid 19', 'melanoma', 'braf']

    for test in tests:
        print()
        for e in entity_tagger.tag_entity(test)[:10]:
            print(e)


if __name__ == "__main__":
    main()
