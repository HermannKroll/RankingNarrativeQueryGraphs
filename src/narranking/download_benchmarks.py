import os.path
import json

import ir_datasets

DATASET_DIR = "/home/kroll/datasets/"

datasets_to_files = [
    ("clinicaltrials/2017/trec-pm-2017", "clinicaltrials_trec-pm-2017",
     ["condition", "summary", "detailed_description", "eligibility"]),

    ("clinicaltrials/2017/trec-pm-2018", "clinicaltrials_trec-pm-2018",
     ["condition", "summary", "detailed_description", "eligibility"]),

    ("clinicaltrials/2019/trec-pm-2019", "clinicaltrials_trec-pm-2019",
     ["condition", "summary", "detailed_description", "eligibility"])

]


def export_dataset_as_json(dataset, json_name, abstract_fields):
    json_file = f'{os.path.join(DATASET_DIR, json_name)}.jsonl'
    if os.path.isfile(json_file):
        print(f'Skip {dataset_name} (already downloaded)')
        return

    with open(json_file, 'wt') as f:
        for idx, doc in enumerate(dataset.docs_iter()):
            abstract = '. \n\n'.join([str(doc._asdict()[field]) for field in abstract_fields])
            json_doc = json.dumps(dict(id=doc.doc_id, title=doc.title, abstract=abstract))
            if idx > 0:
                f.write('\n')
            f.write(json_doc)

    json_query_file = f'{os.path.join(DATASET_DIR, json_name)}_queries.json'
    with open(json_query_file, 'wt') as f:
        queries = []
        for query in dataset.queries_iter():
            queries.append(query._asdict())
        json.dump(queries, f)


for dataset_name, dataset_export_file, abstract_fields in datasets_to_files:
    print(f'Download {dataset_name}...')
    dataset = ir_datasets.load(dataset_name)
    export_dataset_as_json(dataset, dataset_export_file, abstract_fields)
    dataset = None
