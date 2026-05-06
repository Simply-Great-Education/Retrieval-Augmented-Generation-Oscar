from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, CACHE_DIR, BM25_K1, BM25_B, format_search_result

import string
import os
import pickle
import math


from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

class InvertedIndex:
	def __init__(self):
		self.index = defaultdict(set)
		self.docmap: dict[int, dict] = {}
		self.term_frequencies = defaultdict(Counter)
		self.doc_lengths = {}
		self.index_path = os.path.join(CACHE_DIR, "index.pkl")
		self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
		self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
		self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
		
		
	
	def __add_document(self, doc_id, text):
		tokens = tokenization(text)
		for token in set(tokens):
			self.index[token].add(doc_id)
		
		for token in tokens:
			self.term_frequencies[doc_id][token] += 1
		
		self.doc_lengths[doc_id] = len(tokens)
	
	def get_documents(self, term):
		doc_ids = self.index.get(term, set())
		return sorted(list(doc_ids))
	
	def build(self):
		movies = load_movies()
		for movie in movies:
			doc_id = movie["id"]
			self.docmap[doc_id] = movie
			doc_desc = f"{movie['title']} {movie['description']}"
			self.__add_document(doc_id, doc_desc)
	
	def save(self):
		os.makedirs(CACHE_DIR, exist_ok = True)
		with open(self.index_path, 'wb') as f:
			pickle.dump(self.index, f)
		with open(self.docmap_path, 'wb') as f:
			pickle.dump(self.docmap, f)
		with open(self.term_freq_path, 'wb') as f:
			pickle.dump(self.term_frequencies, f)
		with open(self.doc_lengths_path, 'wb') as f:
			pickle.dump(self.doc_lengths, f)
	
	def load(self):
		with open(self.index_path, 'rb') as f:
			self.index = pickle.load(f)
		with open(self.docmap_path, 'rb') as f:
			self.docmap = pickle.load(f)
		with open(self.term_freq_path, 'rb') as f:
			self.term_frequencies = pickle.load(f)
		with open(self.doc_lengths_path, 'rb') as f:
			self.doc_lengths = pickle.load(f)
		
		self._N = len(self.docmap)
		self._avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
	
	def get_tf(self, doc_id, term):
		token = tokenization(term)
		if len(token) != 1:
			raise Exception("Should be one term")

		return self.term_frequencies[doc_id][token[0]]
	
	def get_bm25_idf(self, term: str) -> float:
		token = tokenization(term)
		if len(token) != 1:
			raise Exception("Should be one term")
		N = self._N
		df = len(self.index[token[0]])
		IDF = math.log((N - df + 0.5) / (df + 0.5) + 1)
		return IDF

	def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
		tf = self.get_tf(doc_id, term)
		doc_length = self.doc_lengths.get(doc_id, 0)
		avg_doc_length = self._avg_doc_length
		length_norm = 1 - b + b * (doc_length / avg_doc_length)
		tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
		return tf_component

	
	def __get_avg_doc_length(self) -> float:
		total = 0
		if len(self.doc_lengths) == 0:
			return 0.0
		
		for doc_id in self.doc_lengths:
			total += self.doc_lengths[doc_id]
		
		return total / len(self.doc_lengths)
	
	def bm25(self, doc_id, term):
		tf = self.get_bm25_tf(doc_id, term)
		idf = self.get_bm25_idf(term)
		BM25 = tf * idf
		return BM25
	
	def bm25_search(self, query, limit):
		tokens = tokenization(query)
		scores = {}
		for doc_id in self.docmap:
			total = 0.0
			for token in tokens:
				total += self.bm25(doc_id, token)
			scores[doc_id] = total
		
		sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
		
		results = []
		for doc_id, total in sorted_scores[:limit]:
			doc = self.docmap[doc_id]
			formatted_result = format_search_result(doc_id=doc['id'], title=doc['title'], document=doc['description'], score=total)
			results.append(formatted_result)
		
		return results

def bm25_search_command(query, limit=5):
	index = InvertedIndex()
	index.load()
	return index.bm25_search(query, limit)


def build_command() -> None:

	index = InvertedIndex()

	index.build()

	index.save()

def bm25_idf_command(term):
	index = InvertedIndex()
	index.load()
	bm25_idf = index.get_bm25_idf(term)
	return float(bm25_idf)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
	index = InvertedIndex()
	index.load()
	bm_25_tf = index.get_bm25_tf(doc_id, term, k1, b)
	return bm_25_tf



def idf_command(term):
	token = tokenization(term)
	index = InvertedIndex()
	index.load()
	movies = load_movies()
	total_doc_count = len(movies)
	term_match_doc_count = len(index.index[token[0]])
	idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
	return idf
	

def tf_command(doc_id, term):

	index = InvertedIndex()
	index.load()
	return index.get_tf(doc_id, term)

def tfidf_command(doc_id, term):
	token = tokenization(term)
	index = InvertedIndex()
	index.load()
	movies = load_movies()
	total_doc_count = len(movies)
	term_match_doc_count = len(index.index[token[0]])
	TF = index.get_tf(doc_id, term)
	IDF = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
	tfidf = TF * IDF
	return tfidf


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
	index = InvertedIndex()
	index.load()
	results = []
	seen = set()
	query_tokens = tokenization(query)

	for token in query_tokens:
		for doc_id in index.get_documents(token):
			if doc_id in seen:
				continue
			seen.add(doc_id)
			movie = index.docmap[doc_id]
			results.append(movie)
			if len(results) >= limit:
				return results
				
	return results
	
def preprocess_text(text: str) -> str:
	text = text.lower()
	text = text.translate(str.maketrans("","", string.punctuation))
	return text

def tokenization(text: str) -> list:
	text = preprocess_text(text)
	text = text.split()
	tokens = []
	for token in text:
		if token:
			tokens.append(token)
	
	stop_words = load_stopwords()
	filtered = []
	for token in tokens:
		if token not in stop_words:
			filtered.append(token)
	
	stemmer = PorterStemmer()
	stemmed = []
	for token in filtered:
		stemmed.append(stemmer.stem(token))
		
	return stemmed


def matching_token_check(query_tokens: list, title_tokens: list) -> bool:
	for query_token in query_tokens:
		for title_token in title_tokens:
			if query_token in title_token:
				return True
	return False


	