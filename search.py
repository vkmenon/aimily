import os
import pickle 

import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class HybridSearch:
    def __init__(self, project_name,semantic_weight=0.8, bm25_weight=0.2):
        self.vector_search = SemanticSearch(project_name)
        self.bm25_search = BM25Search(project_name, self.vector_search)
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def add_document(self, doc_id, chunks):
        self.vector_search.add_document(doc_id, chunks)

    def search(self, query, top_k=5):
        vector_results = [((top_k*10)-rank,txt) for rank,txt in enumerate(self.vector_search.search(query, top_k*10))]
        bm25_results = [((top_k*10)-rank,txt) for rank,txt in enumerate(self.bm25_search.search(query, top_k*10))]

        # Create a dictionary to store combined scores
        combined_scores = {}

        # Add BM25 results to the combined scores
        for score, doc in bm25_results:
            if doc in combined_scores:
                combined_scores[doc] += score * self.bm25_weight
            else:
                combined_scores[doc] = score * self.bm25_weight

        # Add vector search results to the combined scores
        for score, doc in vector_results:
            if doc in combined_scores:
                combined_scores[doc] += score * self.semantic_weight
            else:
                combined_scores[doc] = score * self.semantic_weight

        # Sort the combined results based on the combined scores
        combined_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract the top_k results
        combined_results = [c[0] for c in combined_results[:top_k]]
        return combined_results
    
class SemanticSearch:
    def __init__(self, project_name):
        self.db_path = os.path.expanduser(f'~/esg/projects/{project_name}/vector_db.pkl')
        self.db, self.doc_list = self._load_db()
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )

    def _load_db(self):
        if not os.path.exists(self.db_path):
            return dict(), set()
        with open(self.db_path, 'rb') as f:
            return pickle.load(f)

    def _save_db(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump((self.db, self.doc_list), f)

    def _add(self,chunks):
        if isinstance(chunks, str):
            chunks = [chunks]
        embeddings = self.model.encode(chunks)
        print('Building semantic index...')
        for text, emb in tqdm(zip(chunks,embeddings)):
            self.db[tuple(emb)] = text
        print('Done building index!')

    def add_document(self, doc_id, chunks):
        if doc_id in self.doc_list:
            return
        self.doc_list.add(doc_id)
        self._add(chunks)
        self._save_db()

    def search(self, query, top_k=5):
        query_emb = self.model.encode(query, prompt_name='s2p_query')
        doc_embeddings = [np.array(k).flatten() for k in self.db.keys()]
        similarities = self.model.similarity(query_emb, doc_embeddings)
        top_indices = np.argsort(list(similarities.flatten()))[::-1][:top_k]
        return [self.db[tuple(doc_embeddings[i])] for i in top_indices]

class BM25Search:
    def __init__(self, project_name, vector_db):
        self.index_name = project_name
        self.vdb = vector_db

    def _update_bm25_index(self):
        # Update the BM25 index when new documents are added.
        self.corpus = list(self.vdb.db.values())
        self.tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20):
        self._update_bm25_index()
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        # Retrieve the top documents based on their scores
        top_docs = [self.corpus[i] for i in top_indices]
        return top_docs
