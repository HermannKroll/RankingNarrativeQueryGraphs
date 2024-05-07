import pandas as pd
import pyterrier as pt

from narraplay.documentranking.config import DOCUMENT_TEXT_INDEX_PATH

stemmer = "EnglishSnowballStemmer"
stopwords = "none" #"terrier"

if not pt.started():
    pt.init()

pubmed_index = pt.IndexFactory.of(DOCUMENT_TEXT_INDEX_PATH, memory=True)

d_texts = [["0", "Metformin", "PubMed_25212338"],
           ["0", "Metformin", "PubMed_22211893"]]

df = pd.DataFrame(d_texts, columns=["qid", "query", "docno"])
#textscorer = pt.text.scorer(body_attr="text", wmodel="BM25", background_index=pubmed_index,
      #                      properties={'termpipelines' : 'Stopwords,PorterStemmer'})
#rtr = textscorer.transform(df)

pipeline = pt.BatchRetrieve(
    pubmed_index,
    wmodel='BM25',
    properties={'termpipelines' : 'Stopwords,PorterStemmer'}
)

rtr = pipeline(df)

scored_docs = []
for index, row in rtr.iterrows():
    scored_docs.append((row["docno"], max(float(row["score"]), 0.0)))

scored_docs = sorted(scored_docs, key=lambda x: (x[1], x[0]), reverse=True)

print(scored_docs)
