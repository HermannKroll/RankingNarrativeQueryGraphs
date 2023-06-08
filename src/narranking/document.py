from collections import defaultdict
import json
from typing import List

from kgextractiontoolbox.document.document import TaggedDocument, TaggedEntity


class DocumentSentence:

    def __init__(self, sentence_id: str, text: str):
        self.sentence_id = sentence_id
        self.text = text

    def to_dict(self):
        return {"id": self.sentence_id, "text": self.text}


class StatementExtraction:

    def __init__(self, subject_id: str, subject_type: str, subject_str: str,
                 predicate: str, relation: str, object_id: str, object_type: str, object_str: str,
                 sentence_id: int, confidence: float = -1):
        self.subject_id = subject_id
        self.subject_type = subject_type
        self.subject_str = subject_str
        self.predicate = predicate
        self.relation = relation
        self.object_id = object_id
        self.object_type = object_type
        self.object_str = object_str
        self.confidence = confidence
        self.sentence_id = sentence_id

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "subject_str": self.subject_str,
            "predicate": self.predicate,
            "relation": self.relation,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "object_str": self.object_str,
            "sentence_id": self.sentence_id,
            "confidence": self.confidence
        }


class NarrativeDocumentMetadata:

    def __eq__(self, other):
        return self.publication_year == other.publication_year and self.publication_month == other.publication_month \
            and self.authors == other.authors and self.journals == other.journals \
            and self.publication_doi == other.publication_doi

    def __init__(self, publication_year: int, publication_month: int, authors: str, journals: str,
                 publication_doi: str):
        self.publication_year = publication_year
        self.publication_month = publication_month
        self.authors = authors
        self.journals = journals
        self.publication_doi = publication_doi

    def to_dict(self):
        """
       {
          "publication_year":1992,
          "publication_month":4,
          "authors":"Dromer, C | Vedrenne, C | Billey, T | Pages, M | Fourni\u00e9, B | Fourni\u00e9, A",
          "journals":"Revue du rhumatisme et des maladies osteo-articulaires, Vol. 59 No. 4 (Apr 1992)",
          "doi":"https://www.pubpharm.de/vufind/Search/Results?lookfor=NLM1496277"
       }
        :return: a dict with metadata information
        """
        return dict(publication_year=self.publication_year,
                    publication_month=self.publication_month,
                    authors=self.authors,
                    journals=self.journals,
                    doi=self.publication_doi)


class NarrativeDocument(TaggedDocument):

    def __init__(self, document_id: int = None, title: str = None, abstract: str = None,
                 metadata: NarrativeDocumentMetadata = None,
                 tags: List[TaggedEntity] = [],
                 sentences: List[DocumentSentence] = [],
                 extracted_statements: List[StatementExtraction] = []):
        super().__init__(id=document_id, title=title, abstract=abstract, ignore_tags=False)
        self.tags = tags
        if self.tags:
            self.sort_tags()
        self.metadata = metadata
        self.sentences = sentences
        self.extracted_statements = extracted_statements

    def load_from_json(self, json_str: str, ignore_tags=False):
        super().load_from_json(json_str=json_str, ignore_tags=ignore_tags)
        json_dict = json.loads(json_str)
        if 'metadata' in json_dict:
            md = json_dict['metadata']
            self.metadata = NarrativeDocumentMetadata(authors=md.get("authors", None),
                                                      journals=md.get("journals", None),
                                                      publication_doi=md.get("doi", None),
                                                      publication_year=md.get("publication_year", None),
                                                      publication_month=md.get("publication_month", None))

    def to_dict(self):
        """
        {
          "id":1496277,
          "title":"[Rhabdomyolysis due to simvastin. Apropos of a case with review of the literature].",
          "abstract":"A new case of simvastatin-induced acute rhabdomyolysis with heart failure after initiation of treatment with fusidic acid is reported. In most reported instances, statin treatment was initially well tolerated with muscle toxicity developing only after addition of another drug. The mechanism of this muscle toxicity is unelucidated but involvement of a decrease in tissue Co enzyme Q is strongly suspected.",
          "classification": {
                "Pharmaceutical": "drug:drug(356, 360);toxi*:toxicity(305, 313)"
          },
          "tags":[
             {
                "id":"MESH:D012206",
                "mention":"rhabdomyolysis",
                "start":1,
                "end":15,
                "type":"Disease"
             }, ...
          ],
         "metadata":{
              "publication_year":1992,
              "publication_month":4,
              "authors":"Dromer, C | Vedrenne, C | Billey, T | Pages, M | Fourni\u00e9, B | Fourni\u00e9, A",
              "journals":"Revue du rhumatisme et des maladies osteo-articulaires, Vol. 59 No. 4 (Apr 1992)",
              "doi":"https://www.pubpharm.de/vufind/Search/Results?lookfor=NLM1496277"
           }
          "sentences":[
             {
                "id":2456018,
                "text":"A new case of simvastatin-induced acute rhabdomyolysis with heart failure after initiation of treatment with fusidic acid is reported."
             }
          ],
          "statements":[
             {
                "subject_id":"CHEMBL374975",
                "subject_type":"Drug",
                "subject_str":"fusidic acid",
                "predicate":"treatment",
                "relation":"treats",
                "object_id":"MESH:D006333",
                "object_type":"Disease",
                "object_str":"heart failure",
                "sentence_id":2456018
             },
             ...
          ]
        }
        :return:
        """
        tagged_dict = super().to_dict()
        if self.metadata:
            tagged_dict["metadata"] = self.metadata.to_dict()
        if self.sentences:
            tagged_dict["sentences"] = list([s.to_dict() for s in self.sentences])
        if self.extracted_statements:
            tagged_dict["statements"] = list([es.to_dict() for es in self.extracted_statements])

        return tagged_dict

    def __eq__(self, other):
        if not isinstance(other, NarrativeDocument):
            return False
        return self.to_dict() == other.to_dict()


class AnalyzedNarrativeDocument:

    def __init__(self, doc: NarrativeDocument, document_id_art: int, document_id_source: int, collection):
        self.document_id_art = document_id_art
        self.document_id_source = str(document_id_source)
        self.document = doc
        self.collection = collection
        self.concepts = set([t.ent_id for t in doc.tags])
        #    self.concepts.update({t.ent_type for t in doc.tags})
        self.concept2frequency = {}
        for t in doc.tags:
            if t.ent_id not in self.concept2frequency:
                self.concept2frequency[t.ent_id] = 1
            else:
                self.concept2frequency[t.ent_id] += 1
        #         if t.ent_type not in self.concept2frequency:
        #             self.concept2frequency[t.ent_type] = 1
        #         else:
        #             self.concept2frequency[t.ent_type] += 1
        self.concept2statement = None
        self.so2statement = None
        self.statement_concepts = None
        self.objects = None
        self.subjects = None
        self.nodes = None
        self.extracted_statements = None

    def prepare_with_min_confidence(self, min_confidence: float = 0):
        self.subjects = set(
            [s.subject_id for s in self.document.extracted_statements if s.confidence >= min_confidence])
        self.objects = set(
            [s.object_id for s in self.document.extracted_statements if s.confidence >= min_confidence])
        self.statement_concepts = set(
            [(s.subject_id, s.object_id) for s in self.document.extracted_statements if s.confidence >= min_confidence])
        self.statement_concepts.update(
            [(s.object_id, s.subject_id) for s in self.document.extracted_statements if s.confidence >= min_confidence])

        self.nodes = self.subjects.union(self.objects)
        self.so2statement = defaultdict(list)
        self.concept2statement = defaultdict(list)
        for statement in filter(lambda s: s.confidence >= min_confidence, self.document.extracted_statements):
            self.so2statement[(statement.subject_id, statement.object_id)].append(statement)
            self.so2statement[(statement.object_id, statement.subject_id)].append(statement)

            self.concept2statement[statement.subject_id].append(statement)
            self.concept2statement[statement.object_id].append(statement)

        self.extracted_statements = list([s for s in self.document.extracted_statements
                                          if s.confidence >= min_confidence])

    def get_length_in_words(self):
        text = self.get_text()
        return len(text.split(' '))

    def get_length_in_concepts(self):
        count = 0
        for c, freq in self.concept2frequency.items():
            count += freq
        return count

    def get_concept_frequency(self, concept):
        if concept in self.concept2frequency:
            return self.concept2frequency[concept]
        else:
            return 0

    def get_text(self):
        return self.document.get_text_content(sections=True)

    def to_dict(self):
        return {"document": self.document.to_dict(),
                "concepts": str(self.concepts),
                "concept2frequency": self.concept2frequency}
