import itertools
from typing import List, Set

import networkx

from narranking.corpus import DocumentCorpus
from narranking.document import AnalyzedNarrativeDocument
from narranking.query import AnalyzedQuery
from narranking.rankers.abstract import AbstractDocumentRanker


class GraphConnectivityDocumentRanker:

    def __init__(self):
        pass

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        pass


class PathFrequencyRanker(AbstractDocumentRanker):
    def __init__(self, name="PathFrequencyRanker"):
        super().__init__(name=name)

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        doc_scores = []

        for doc in narrative_documents:
            graph = self._document_graph(doc)
            stmts = list(self._concept_product(query, doc))
            paths = 0
            for subj, obj in stmts:
                # do not search for self combinations
                if subj == obj:
                    continue
                # skip if one of the values does not exist as a node
                if subj not in graph or obj not in graph:
                    continue
                # count existing paths for a statement and remove them lengthwise starting with the shortest possible.
                while networkx.has_path(graph, subj, obj):
                    shortest_path = networkx.shortest_path(graph, subj, obj)
                    for i in range(len(shortest_path) - 1):
                        graph.remove_edge(shortest_path[i], shortest_path[i + 1])
                    paths += self._evaluate_path_score(doc, shortest_path)
            doc_scores.append((doc.document_id_source, paths))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        return len(shortest_path)

    @staticmethod
    def _document_graph(doc: AnalyzedNarrativeDocument) -> networkx.MultiGraph:
        graph: networkx.MultiGraph = networkx.MultiGraph()
        for stmt in doc.extracted_statements:
            graph.add_edge(stmt.subject_id, stmt.object_id)
        return graph

    @staticmethod
    def _concept_product(query: AnalyzedQuery, doc: AnalyzedNarrativeDocument):
        concepts: Set[str] = doc.concepts.intersection(query.concepts)
        return itertools.product(concepts, concepts)


class PathFrequencyRankerWeightedOld(PathFrequencyRanker):
    def __init__(self, name="PathFrequencyRankerWeightedOld"):
        super().__init__(name=name)

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        partial_concept_pairs = list(itertools.combinations(query.partial_concepts, 2))
        partial_concept_pairs_idx = list(itertools.combinations([*range(len(query.partial_concepts))], 2))
        max_scores = [0] * len(query.partial_concepts)

        doc_scores = []
        for doc in narrative_documents:
            graph = self._document_graph(doc)
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = self._partial_concept_product(doc, c1, c2)
                paths = 0
                for subj, obj in query_concept_pairs:
                    # do not search for self combinations
                    if subj == obj:
                        continue
                    # skip if one of the values does not exist as a node
                    if subj not in graph or obj not in graph:
                        continue
                    # count existing paths for a statement and remove them lengthwise starting with the shortest possible.
                    while networkx.has_path(graph, subj, obj):
                        shortest_path = networkx.shortest_path(graph, subj, obj)
                        for i in range(len(shortest_path) - 1):
                            graph.remove_edge(shortest_path[i], shortest_path[i + 1])
                        paths += self._evaluate_path_score(doc, shortest_path)
                max_scores[i1] = max(max_scores[i1], paths)
                max_scores[i2] = max(max_scores[i2], paths)

        for doc in narrative_documents:
            graph = self._document_graph(doc)
            score = 0
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = self._partial_concept_product(doc, c1, c2)
                paths = 0
                im_score = 0
                for subj, obj in query_concept_pairs:
                    # do not search for self combinations
                    if subj == obj:
                        continue
                    # skip if one of the values does not exist as a node
                    if subj not in graph or obj not in graph:
                        continue
                    # count existing paths for a statement and remove them lengthwise starting with the shortest possible.
                    while networkx.has_path(graph, subj, obj):
                        shortest_path = networkx.shortest_path(graph, subj, obj)
                        for i in range(len(shortest_path) - 1):
                            graph.remove_edge(shortest_path[i], shortest_path[i + 1])
                        paths += self._evaluate_path_score(doc, shortest_path)
                if max_scores[i1] != 0:
                    im_score += paths / max_scores[i1]
                if max_scores[i2] != 0:
                    im_score += paths / max_scores[i2]
                score += im_score / 2
            doc_scores.append((doc.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _partial_concept_product(doc: AnalyzedNarrativeDocument, concepts_lhs: Set[str], concepts_rhs: Set[str]):
        concepts_lhs: Set[str] = doc.concepts.intersection(concepts_lhs)
        concepts_rhs: Set[str] = doc.concepts.intersection(concepts_rhs)
        return list(itertools.product(concepts_lhs, concepts_rhs))


class PathAbstractRankerWeighted(PathFrequencyRanker):
    def __init__(self, name="PathAbstractRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        raise NotImplementedError()

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        partial_concept_pairs = list(itertools.combinations(query.partial_concepts, 2))
        partial_concept_pairs_idx = list(itertools.combinations([*range(len(query.partial_concepts))], 2))
        max_scores = [0] * len(query.partial_concepts)

        doc_scores = []
        for doc in narrative_documents:
            graph = self._document_graph(doc)
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = self._partial_concept_product(doc, c1, c2)
                paths = 0
                for subj, obj in query_concept_pairs:
                    # do not search for self combinations
                    if subj == obj:
                        continue
                    # skip if one of the values does not exist as a node
                    if subj not in graph or obj not in graph:
                        continue
                    # count existing paths for a statement and remove them lengthwise starting with the shortest possible.
                    while networkx.has_path(graph, subj, obj):
                        shortest_path = networkx.shortest_path(graph, subj, obj)
                        for i in range(len(shortest_path) - 1):
                            graph.remove_edge(shortest_path[i], shortest_path[i + 1])
                        paths += self._evaluate_path_score(doc, shortest_path)
                max_scores[i1] = max(max_scores[i1], paths)
                max_scores[i2] = max(max_scores[i2], paths)

        for doc in narrative_documents:
            graph = self._document_graph(doc)
            score = 0
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = self._partial_concept_product(doc, c1, c2)
                paths = 0
                im_score = 0
                for subj, obj in query_concept_pairs:
                    # do not search for self combinations
                    if subj == obj:
                        continue
                    # skip if one of the values does not exist as a node
                    if subj not in graph or obj not in graph:
                        continue
                    # count existing paths for a statement and remove them lengthwise starting with the shortest possible.
                    while networkx.has_path(graph, subj, obj):
                        shortest_path = networkx.shortest_path(graph, subj, obj)
                        for i in range(len(shortest_path) - 1):
                            graph.remove_edge(shortest_path[i], shortest_path[i + 1])
                        paths += self._evaluate_path_score(doc, shortest_path)
                if max_scores[i1] != 0:
                    im_score += paths / max_scores[i1]
                if max_scores[i2] != 0:
                    im_score += paths / max_scores[i2]
                score += im_score / 2

            score += 1.0 / len(partial_concept_pairs)
            doc_scores.append((doc.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _partial_concept_product(doc: AnalyzedNarrativeDocument, concepts_lhs: Set[str], concepts_rhs: Set[str]):
        concepts_lhs: Set[str] = doc.nodes.intersection(concepts_lhs)
        concepts_rhs: Set[str] = doc.nodes.intersection(concepts_rhs)
        return list(itertools.product(concepts_lhs, concepts_rhs))


class PathsFrequencyRankerWeighted(PathAbstractRankerWeighted):
    def __init__(self, name="PathsFrequencyRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        return 1.0


class PathsFrequencyInverseLengthRankerWeighted(PathAbstractRankerWeighted):
    def __init__(self, name="PathsFrequencyInverseLengthRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        return 1.0 / len(shortest_path)


class PathsFrequencyLengthRankerWeighted(PathAbstractRankerWeighted):
    def __init__(self, name="PathsFrequencyLengthRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        return len(shortest_path)


class ShortestPathAbstractRankerWeighted(PathFrequencyRanker):
    def __init__(self, name="ShortestPathAbstractRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        raise NotImplementedError()

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        partial_concept_pairs = list(itertools.combinations(query.partial_concepts, 2))
        partial_concept_pairs_idx = list(itertools.combinations([*range(len(query.partial_concepts))], 2))
        max_scores = [0] * len(query.partial_concepts)

        doc_scores = []
        for doc in narrative_documents:
            graph = self._document_graph(doc)
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = self._partial_concept_product(doc, c1, c2)
                paths = 0
                for subj, obj in query_concept_pairs:
                    # do not search for self combinations
                    if subj == obj:
                        continue
                    # skip if one of the values does not exist as a node
                    if subj not in graph or obj not in graph:
                        continue

                    # get the shortest path
                    if networkx.has_path(graph, subj, obj):
                        shortest_path = networkx.shortest_path(graph, subj, obj)
                        paths += self._evaluate_path_score(doc, shortest_path)
                max_scores[i1] = max(max_scores[i1], paths)
                max_scores[i2] = max(max_scores[i2], paths)

        for doc in narrative_documents:
            graph = self._document_graph(doc)
            score = 0
            for (i1, i2), (c1, c2) in zip(partial_concept_pairs_idx, partial_concept_pairs):
                query_concept_pairs = self._partial_concept_product(doc, c1, c2)
                paths = 0
                im_score = 0
                for subj, obj in query_concept_pairs:
                    # do not search for self combinations
                    if subj == obj:
                        continue
                    # skip if one of the values does not exist as a node
                    if subj not in graph or obj not in graph:
                        continue

                    # get the shortest path
                    if networkx.has_path(graph, subj, obj):
                        shortest_path = networkx.shortest_path(graph, subj, obj)
                        paths += self._evaluate_path_score(doc, shortest_path)
                if max_scores[i1] != 0:
                    im_score += paths / max_scores[i1]
                if max_scores[i2] != 0:
                    im_score += paths / max_scores[i2]
                score += im_score / 2

            score += 1.0 / len(partial_concept_pairs)
            doc_scores.append((doc.document_id_source, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _partial_concept_product(doc: AnalyzedNarrativeDocument, concepts_lhs: Set[str], concepts_rhs: Set[str]):
        concepts_lhs: Set[str] = doc.nodes.intersection(concepts_lhs)
        concepts_rhs: Set[str] = doc.nodes.intersection(concepts_rhs)
        return list(itertools.product(concepts_lhs, concepts_rhs))


class ShortestPathLengthRankerWeighted(ShortestPathAbstractRankerWeighted):
    def __init__(self, name="ShortestPathLengthRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        return len(shortest_path)


class ShortestPathInverseLengthRankerWeighted(ShortestPathAbstractRankerWeighted):
    def __init__(self, name="ShortestPathInverseLengthRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        return 1.0 / len(shortest_path)


class ShortestPathFrequencyRankerWeighted(ShortestPathAbstractRankerWeighted):
    def __init__(self, name="ShortestPathFrequencyRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        score = 0
        for i in range(len(shortest_path) - 1):
            subj, obj = shortest_path[i], shortest_path[i + 1]
            if (subj, obj) in doc.so2statement:
                for s in doc.so2statement[subj, obj]:
                    score += 1.0
        return score


class ShortestPathConfidenceRankerWeighted(ShortestPathAbstractRankerWeighted):
    def __init__(self, name="ShortestPathConfidenceRankerWeighted"):
        super().__init__(name=name)

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, shortest_path: [str]) -> float:
        score = 0
        for i in range(len(shortest_path) - 1):
            subj, obj = shortest_path[i], shortest_path[i + 1]
            if (subj, obj) in doc.so2statement:
                for s in doc.so2statement[subj, obj]:
                    score += s.confidence
        return score


class ConfidencePathFrequencyRanker(PathFrequencyRanker):
    def __init__(self):
        super().__init__(name="ConfidencePathFrequencyRanker")

    @staticmethod
    def _evaluate_path_score(doc: AnalyzedNarrativeDocument, path: [str]) -> float:
        score = 0

        for i in range(len(path) - 1):
            subj = path[i]
            obj = path[i + 1]

            for stmt in doc.so2statement[(subj, obj)]:
                score += stmt.confidence
        return score


class ConfidencePathFrequencyRankerWeighted(PathFrequencyRankerWeightedOld, ConfidencePathFrequencyRanker):
    def __init__(self):
        super(PathFrequencyRankerWeightedOld, self).__init__()
        super(ConfidencePathFrequencyRanker, self).__init__(name="ConfidencePathFrequencyRankerWeighted")

    def rank_documents(self, query: AnalyzedQuery, narrative_documents: List[AnalyzedNarrativeDocument],
                       corpus: DocumentCorpus):
        return super(PathFrequencyRankerWeightedOld, self).rank_documents(query, narrative_documents, corpus)

    def _evaluate_path_score(self, doc: AnalyzedNarrativeDocument, path: [str]) -> float:
        return super(ConfidencePathFrequencyRanker, self)._evaluate_path_score(doc, path)
