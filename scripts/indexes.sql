-- must be executed as superuser
CREATE EXTENSION pg_trgm;

CREATE INDEX index_entity_tagger_data_synonym ON entity_tagger_data USING gin (synonym gin_trgm_ops);