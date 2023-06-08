# Graph-based Ranking Methods for Biomedical Document Retrieval
This repository belongs to our TPDL2023 submission. 
Thank you for being here. 
This repository is still updated.

Please note that the repository has a dependency to our [Narrative Service](www.narrative.pubpharm.de) whose code is not publicly available. 
We are working to integrate all code that is required into this repository. 
Still, we cannot publish the pre-processed document data (the extracted graph information).

# Documentation
The implemented code is available in the [src](src) directory.
It contains:
- [Benchmark processing](src/narranking/benchmark.py)
- [BM25 baseline](src/narranking/create_bm25_baseline.py)
- [Graph-based Ranking Methods](src/narranking/rankers)
- [Running the experiments](src/narranking/main.py)

The evaluation reports for all benchmarks (TREC Precision Medicine 2017, 2018, 2019 and 2020) are available in our [evaluation](evaluation) directory.


# Setup

## Subrepository: KGExtractionToolbox 
This project builds upon our extraction toolbox.
That is why we also need to check out its code.
- [KGExtractionToolbox](https://github.com/HermannKroll/KGExtractionToolbox): Basic entity linking methods / information extraction / pipelines for automatisation

Please read the toolbox setup documentation. 
However, we cannot share the actual document data.

To use this project, clone this project and its submodules via:
```
git clone --recurse-submodules git@github.com:HermannKroll/RankingNarrativeQueryGraphs.git
```



## Setup Python Environment
We used a Conda Environment and Python 3.8
```
conda create -n ranking python=3.8
```

Install the requirements of this project and of the toolbox via:
```
pip install -r requirements.txt
pip install -r lib/KGExtractionToolbox/requirements.txt
```

Set the Python path:
```
export PYTHONPATH="/home/USER/RankingNarrativeQueryGraphs/src/:/home/USER/RankingNarrativeQueryGraphs/lib/KGExtractionToolbox/src/"
```