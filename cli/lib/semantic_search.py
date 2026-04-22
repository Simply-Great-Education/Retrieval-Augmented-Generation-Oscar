
from .search_utils import load_movies, DEFAULT_CHUNK_SIZE

from sentence_transformers import SentenceTransformer

import numpy as np
import os
import re


class SemanticSearch:
	def __init__(self):
		self.model = SentenceTransformer('all-MiniLM-L6-v2')
		self.embeddings = None
		self.documents = None
		self.document_map = {}
	
	
	def generate_embedding(self, text):
		if not text or not text.strip():
			raise ValueError("Text input is empty")

		return self.model.encode([text])[0]
	
	def build_embeddings(self, documents):
		self.documents = documents
		doc_list = []
		for doc in documents:
			doc_id = doc["id"]
			self.document_map[doc_id] = doc
			doc_list.append(f"{doc['title']}: {doc['description']}")
		
		self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
		
		numpy_file = "cache/movie_embeddings.npy"
		
		np.save(numpy_file, self.embeddings)
		
		return self.embeddings
	
	def load_or_create_embeddings(self, documents):
		self.documents = documents
		doc_list = []
		for doc in documents:
			doc_id = doc["id"]
			self.document_map[doc_id] = doc
			doc_list.append(f"{doc['title']}: {doc['description']}")
		
		numpy_file = "cache/movie_embeddings.npy"
		
		if os.path.exists(numpy_file):
			self.embeddings = np.load(numpy_file)
			
			if len(self.embeddings) == len(documents):
				return self.embeddings
		
		return self.build_embeddings(documents)
	
	def search(self, query, limit):
		if self.documents is None:
			raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
		
		query_embedding = self.generate_embedding(query)
		
		cosine_scores = []
		
		for i, doc_embedding in enumerate(self.embeddings):
			score = cosine_similarity(query_embedding, doc_embedding)
			cosine_scores.append((score, self.documents[i]))
		
		sorted_scores = sorted(cosine_scores, key=lambda pair: pair[0], reverse=True)
		
		top_scores = []
		
		for score, doc in sorted_scores[:limit]:
			item = {'score': score, 'title': doc["title"], 'description': doc["description"]}
			top_scores.append(item)


		return top_scores

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
       
    def build_chunk_embeddings(self, documents) # work in progress
		self.documents = documents
		doc_list = []
		for doc in documents:
			doc_id = doc["id"]
			self.document_map[doc_id] = doc
			doc_list.append(f"{doc['title']}: {doc['description']}")
		
		self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
		
		numpy_file = "cache/chunk_embeddings.npy"
		
		np.save(numpy_file, self.embeddings)
		
		return self.embeddings

	
def verify_model():
	ss = SemanticSearch()

	print(f"Model loaded: {ss.model}")
	print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text):
	ss = SemanticSearch()
	
	embedding = ss.generate_embedding(text)
	
	print(f"Text: {text}")
	print(f"First 3 dimensions: {embedding[:3]}")
	print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
	ss = SemanticSearch()
	
	documents = load_movies()
	
	embeddings = ss.load_or_create_embeddings(documents)
	
	print(f"Number of docs:   {len(documents)}")
	print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
	ss = SemanticSearch()
	
	embedding = ss.generate_embedding(query)
	
	print(f"Query: {query}")
	print(f"First 3 dimensions: {embedding[:3]}")
	print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
	dot_product = np.dot(vec1, vec2)
	norm1 = np.linalg.norm(vec1)
	norm2 = np.linalg.norm(vec2)

	if norm1 == 0 or norm2 == 0:
		return 0.0

	return dot_product / (norm1 * norm2)



def search_command(query, limit):
	ss = SemanticSearch()
	
	documents = load_movies()
	
	embeddings = ss.load_or_create_embeddings(documents)
	
	results = ss.search(query, limit)
	
	for i, doc in enumerate(results, 1):
		print(f"{i}. {doc['title']} (score:{doc['score']:.4f})")
		print(f"{doc['description'][:100]}...")
		print()

def chunk_text(text, size, overlap):
	chunks = size_chunking(text, size, overlap)
	print(f"Chunking {len(text)} characters")
	for i, chunk in enumerate(chunks):
		print(f"{i + 1}. {chunk}")

def semantic_chunk_text(text, size, overlap):
	chunks = semantic_size_chunking(text, size, overlap)
	print(f"Semantically chunking {len(text)} characters")
	for i, chunk in enumerate(chunks):
		print(f"{i + 1}. {chunk}")


def size_chunking(text, size, overlap):
	words = text.split()
	chunks = []
	i = 0
	while i < len(words):
		chunk = words[i:i+size]
		if chunks and len(chunk) <= overlap:
			break
			
		chunks.append(" ".join(chunk))
		i += size - overlap
	
	return chunks

def semantic_size_chunking(text, size, overlap):
	words = re.split(r"(?<=[.!?])\s+", text)
	chunks = []
	i = 0
	while i < len(words):
		chunk = words[i:i+size]
		if chunks and len(chunk) <= overlap:
			break
			
		chunks.append(" ".join(chunk))
		i += size - overlap
	
	return chunks
























