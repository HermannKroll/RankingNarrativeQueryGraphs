# Ranking Narrative Query Graphs for Biomedical Document Retrieval
This repository belongs to our workshop publication at [AI4LAC@JCDL2024](https://zhanghaoxuan1999.github.io/JCDL2024-AI4LAC-workshop/). You find a short version of our article on the workshop webpage and a technical report (long version) at arXiv. The links will be added as soon as they are available. 

This repository contains graph-based ranking methods for our Narrative Service.
Please note that the repository has a dependency to our [Narrative Service](www.narrative.pubpharm.de) whose code is publicly available on [GitHub](https://github.com/HermannKroll/NarrativeIntelligence). 
Unfortunately, we cannot publish the pre-processed document data (the extracted graph information).

# Documentation
The implemented code is available in the [src](src) directory.
It contains:
- [Benchmark processing](src/narranking/benchmark.py)
- [BM25 baseline](src/narranking/baselines/create_bm25_baseline.py)
- [Graph-based Ranking Methods](src/narranking/rankers)
- [Running the experiments](src/narranking/main.py)

The evaluation reports for all benchmarks (TREC Precision Medicine 2017, 2018, 2019, 2020 and TREC Covid) are available 
in our [evaluation](evaluation) directory.


# Setup

## Sub-repositories: KGExtractionToolbox, NarrativeAnnotation, NarrativeIntelligence
This project builds upon our previously published projects. 
That is why we also need to check out their code.
Currently, we rely on the dev branches.
- [KGExtractionToolbox](https://github.com/HermannKroll/KGExtractionToolbox/tree/dev): Basic entity linking methods / information extraction / pipelines for automation
- [NarrativeAnnotation](https://github.com/HermannKroll/NarrativeAnnotation/tree/dev): Pharmaceutical specific entity linking / text classification / statement extraction logic
- [NarrativeIntelligence](https://github.com/HermannKroll/NarrativeIntelligence/tree/dev): Code and scripts for PubPharm's Narrative Service

Please read the toolbox setup documentation.

To use this project, clone this project and its submodules via:
```
git clone --recurse-submodules git@github.com:HermannKroll/RankingNarrativeQueryGraphs.git
```



## Setup Python Environment
We used a Conda Environment and Python 3.8
```
conda create -n ranking python=3.8
```

Install the requirements of this project and of the submodules via:
```
pip install -r requirements.txt
pip install -r lib/KGExtractionToolbox/requirements.txt
pip install -r lib/NarrativeIntelligence/requirements.txt
pip install -r lib/NarrativeAnnotation/requirements.txt
```

Set the Python path:
```
export PYTHONPATH="/home/USER/RankingNarrativeQueryGraphs/src/:/home/USER/RankingNarrativeQueryGraphs/lib/KGExtractionToolbox/src/:/home/USER/RankingNarrativeQueryGraphs/lib/NarrativeAnnotation/src/:/home/USER/RankingNarrativeQueryGraphs/lib/NarrativeIntelligence/src/"
```
