#!/bin/bash

VACUUM_SQL="/home/kroll/NarrativeIntelligence/sql/vacuum_db.sql"
PREDICATION_CLEANING_SQL="/home/kroll/NarrativeIntelligence/lib/NarrativeAnnotation/sql/clean_predication.sql"


bash process_documents.sh /home/kroll/datasets/clinicaltrials_trec-pm-2017.jsonl trec-pm-2017-cds
bash process_documents.sh /home/kroll/datasets/clinicaltrials_trec-pm-2018.jsonl trec-pm-2018-cds
bash process_documents.sh /home/kroll/datasets/clinicaltrials_trec-pm-2019.jsonl trec-pm-2019-cds
bash process_documents.sh /home/kroll/datasets/pm2017_extra_documents.json trec-pm-201X-extra
bash process_documents.sh /home/kroll/datasets/trec_covid_round5_abstract.json trec_covid_round5_abstract
bash process_documents.sh /home/kroll/datasets/trec_covid_round5_abstract_fulltext.json trec_covid_round5_abstract_fulltext
bash process_documents.sh /home/kroll/datasets/trip_click_dataset.json trip_click_dataset



# Do the canonicalizing step & Apply the rules
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trec-pm-2017-cds  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trec-pm-2018-cds  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trec-pm-2019-cds  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trec-pm-201X-extra  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trec_covid_round5_abstract  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json

python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trec_covid_round5_abstract_fulltext  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trip_click_dataset  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json




python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trec-pm-2017-cds
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trec-pm-2018-cds
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trec-pm-2019-cds
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trec-pm-201X-extra
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trec_covid_round5_abstract

python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trec_covid_round5_abstract_fulltext
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trip_click_dataset


python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c PubMed --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c PubMed

# Execute Cleaning Rules for Predications
echo 'cleaning predication table with hand-written rules'
# execute general rules
psql "host=127.0.0.1 port=5432 dbname=cikm2023 user=tagginguser password=u3j4io1234u8-13!14" -f $PREDICATION_CLEANING_SQL
# apply special PathIE and COS cleaning
psql "host=127.0.0.1 port=5432 dbname=cikm2023 user=tagginguser password=u3j4io1234u8-13!14" -f predicaton_cleaning.sql

# Finally vacuum all tables
# Execute Cleaning Rules for Tagging
echo 'vacuum db tables...'
psql "host=127.0.0.1 port=5432 dbname=cikm2023 user=tagginguser password=u3j4io1234u8-13!14" -f $VACUUM_SQL

python3 ~/NarrativeIntelligence/src/narraint/queryengine/index/compute_reverse_index_predication.py
python3 ~/NarrativeIntelligence/src/narraint/queryengine/index/compute_reverse_index_tag.py


# If you want to delete the collections again
python3 ~/NarrativeIntelligence/src/narraint/backend/delete_collection.py trec-pm-2017-cds
python3 ~/NarrativeIntelligence/src/narraint/backend/delete_collection.py trec-pm-2018-cds
python3 ~/NarrativeIntelligence/src/narraint/backend/delete_collection.py trec-pm-2019-cds
python3 ~/NarrativeIntelligence/src/narraint/backend/delete_collection.py trec-pm-201X-extra

python3 ~/NarrativeIntelligence/src/narraint/backend/delete_collection.py trec_covid_round5_abstract
python3 ~/NarrativeIntelligence/src/narraint/backend/delete_collection.py trec_covid_round5_abstract_fulltext
python3 ~/NarrativeIntelligence/src/narraint/backend/delete_collection.py trip_click_dataset