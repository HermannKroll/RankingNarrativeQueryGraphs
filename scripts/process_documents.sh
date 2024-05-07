#!/bin/bash
# $1 - document path
# $2 - document collection
TAG_CLEANING_SQL="/home/kroll/NarrativeIntelligence/lib/NarrativeAnnotation/sql/clean_tags.sql"
TEMP_GNORMPLUS="/home/kroll/datasets/gnormplus_temp.json"


echo $1
echo $2

## Load everything
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/document/load_document.py $1 -c $2 --artificial_document_ids

if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi

# Next, tag the documents with our PharmDictTagger
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/entitylinking/dictpreprocess.py -c $2 --workers 32 --sections

if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi

# Export the document content
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/document/export.py -d $TEMP_GNORMPLUS --collection $2 --format json

if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi

# Run GNormPlus
python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/entitylinking/biomedical_entity_linking.py $TEMP_GNORMPLUS -c $2 --skip-load --workers 15 --gnormplus

if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi


# Some gene annotations are composed (e.g, id = 123;345) this ids need to be split into multiple tag entries
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/clean_tag_gene_ids.py
if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi


# Execute Cleaning Rules for Tagging
echo 'cleaning Tag table with hand-written rules'
psql "host=127.0.0.1 port=5432 dbname=cikm2023 user=tagginguser password=u3j4io1234u8-13!14" -f $TAG_CLEANING_SQL


if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi

# Do the statement extraction for all $2 documents via our Pipeline
python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/extraction/pharmaceutical_pipeline.py -c $2 -et PathIE --workers 26 --relation_vocab /home/kroll/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json --sections


if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi


python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/extraction/pharmaceutical_pipeline.py -c $2 -et COSentence --workers 50 -bs 1000000


if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi

