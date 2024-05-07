
bash process_documents.sh /home/kroll/datasets/clinicaltrials_trec-pm-2017.jsonl trec-pm-2017-cds


python3 ~/NarrativeIntelligence/lib/KGExtractionToolbox/src/kgextractiontoolbox/cleaning/canonicalize_predicates.py -c trec-pm-2017-cds  --word2vec_model /home/kroll/workingdir/BioWordVec_PubMed_MIMICIII_d200.bin --relation_vocab ~/NarrativeIntelligence/lib/NarrativeAnnotation/resources/pharm_relation_vocab.json

if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi

python3 ~/NarrativeIntelligence/lib/NarrativeAnnotation/src/narrant/cleaning/pharmaceutical_rules.py -c trec-pm-2017-cds

if [[ $? != 0 ]]; then
    echo "Previous script returned exit code != 0 -> Stopping pipeline."
    exit -1
fi

