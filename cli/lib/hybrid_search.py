import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import format_search_result, load_movies
from .query_enhancement import enhance_query
from .reranking import rerank

class HybridSearch:
	def __init__(self, documents):
		self.documents = documents
		self.semantic_search = ChunkedSemanticSearch()
		self.semantic_search.load_or_create_chunk_embeddings(documents)

		self.idx = InvertedIndex()
		if not os.path.exists(self.idx.index_path):
			self.idx.build()
			self.idx.save()

	def _bm25_search(self, query, limit):
		self.idx.load()
		return self.idx.bm25_search(query, limit)

	def weighted_search(self, query, alpha, limit=5):
		bm25_results = self._bm25_search(query, (limit * 500))
		
		semantic_results = self.semantic_search.search_chunks(query, (limit * 500))
		
		raw_bm25 = [r["score"] for r in bm25_results]
		normalized_bm25 = normalize_scores(raw_bm25)
		
		for r, n in zip(bm25_results, normalized_bm25):
			r["normalized_score"] = n
		
		raw_semantic = [r["score"] for r in semantic_results]
		normalized_semantic = normalize_scores(raw_semantic)
		
		for r, n in zip(semantic_results, normalized_semantic):
			r["normalized_score"] = n
		
		
		combined = combine_search_results(bm25_results, semantic_results, alpha)
		
		return combined[:limit]

	def rrf_search(self, query, k, limit=10):
		bm25_results = self._bm25_search(query, (limit * 500))
		semantic_results = self.semantic_search.search_chunks(query, (limit * 500))
		
		rrf_results = {}
		
		for i, item in enumerate(bm25_results):
			doc_id = item["id"]
			if doc_id not in rrf_results:
				rrf_results[doc_id] = {
					
					"title": item["title"],
					"document": item["document"],
					"bm25_rank": 10000,
					"semantic_rank": 10000,
				}
			
			if rrf_results[doc_id]["bm25_rank"] == 10000:
				rrf_results[doc_id]["bm25_rank"] = i + 1
		
		for i, item in enumerate(semantic_results):
			doc_id = item["id"]
			if doc_id not in rrf_results:
				rrf_results[doc_id] = {
					
					"title": item["title"],
					"document": item["document"],
					"bm25_rank": 10000,
					"semantic_rank": 10000,
				}
			
			if rrf_results[doc_id]["semantic_rank"] == 10000:
				rrf_results[doc_id]["semantic_rank"] = i + 1
		
		results = []

		for doc_id, ranks in rrf_results.items():
			bm25_rrf = rrf_score(ranks["bm25_rank"], k)
			semantic_rrf = rrf_score(ranks["semantic_rank"], k)
			combined_rrf_score = bm25_rrf + semantic_rrf
			result = format_search_result(
				doc_id=doc_id,
				title=ranks["title"],
				document=ranks["document"],
				score=combined_rrf_score,
				bm25_rank=ranks["bm25_rank"],
				semantic_rank=ranks["semantic_rank"],
			)

			results.append(result)


		sorted_result = sorted(results, key=lambda x: x["score"], reverse=True)
		
		return sorted_result[:limit]

		

def normalize_scores(scores):
	if scores == None:
		return []
		
	small = min(scores)
	large = max(scores)
	
	if small == large:
		return [1.0] * len(scores)
	
	results = []
	for num in scores:
		score = (num - small) / (large - small)
		results.append(score)
	
	return results

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
	return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):

	return 1 / (k + rank)	


def combine_search_results(bm25_scores, semantic_scores, alpha):

	combined_scores = {}


	for item in bm25_scores:

		doc_id = item["id"]

		if doc_id not in combined_scores:

			combined_scores[doc_id] = {

				"title": item["title"],

				"document": item["document"],

				"bm25_score": 0.0,

				"semantic_score": 0.0,

			}

		if item["normalized_score"] > combined_scores[doc_id]["bm25_score"]:

			combined_scores[doc_id]["bm25_score"] = item["normalized_score"]


	for item in semantic_scores:

		doc_id = item["id"]

		if doc_id not in combined_scores:

			combined_scores[doc_id] = {

				"title": item["title"],

				"document": item["document"],

				"bm25_score": 0.0,

				"semantic_score": 0.0,

			}

		if item["normalized_score"] > combined_scores[doc_id]["semantic_score"]:

			combined_scores[doc_id]["semantic_score"] = item["normalized_score"]

	results = []

	for doc_id, scores in combined_scores.items():

		hybrid_score_val = hybrid_score(scores["bm25_score"], scores["semantic_score"], alpha)

		result = format_search_result(

			doc_id=doc_id,

			title=scores["title"],

			document=scores["document"],

			score=hybrid_score_val,

			bm25_score=scores["bm25_score"],

			semantic_score=scores["semantic_score"],

		)

		results.append(result)



	return sorted(results, key=lambda x: x["score"], reverse=True)



def weighted_search_command(query, alpha, limit):
	movies = load_movies()
	hs = HybridSearch(movies)
	
	original_query = query
	
	search_results = hs.weighted_search(query, alpha, limit)
	
	return {

		"original_query": original_query,

		"query": query,

		"alpha": alpha,

		"results": search_results,

	}
	
def rrf_search_command(query, k, enhance, limit, rerank_method):
	movies = load_movies()
	hs = HybridSearch(movies)
	
	original_query = query
	enhanced_query = None
	
	if enhance:
		enhanced_query = enhance_query(query, method=enhance)
		query = enhanced_query
	
	print(f"enhanced query - {query}")
	
	search_limit = limit
	
	if rerank_method:
		search_limit = limit * 5
	
		
	search_results = hs.rrf_search(query, k, search_limit)
	
	for doc in search_results:
		print(f"{doc['title']}, {doc['score']}")
	
	reranked = False
	if rerank_method:
		search_results = rerank(query, search_results, rerank_method, limit)
		reranked = True
	
	
	return {

		"original_query": original_query,
		
		"enhanced_query": enhanced_query,
		
		"enhance_method": enhance,

		"query": query,

		"k": k,
		
		"rerank_method": rerank_method,
		
		"reranked": reranked,

		"results": search_results,

	}



def weighted_search_results(query, alpha, limit):
	
	result = weighted_search_command(query, alpha, limit)
	
	print(f"Weighted Hybrid Search Results for '{result['query']}':")

	print(f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic")
	
	print()
	
	for i, res in enumerate(result["results"], 1):

		print(f"{i}. {res['title']}")

		print(f"   Hybrid Score: {res.get('score', 0):.3f}")

		metadata = res.get("metadata", {})

		if "bm25_score" in metadata and "semantic_score" in metadata:

			print(f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}")

			print(f"   {res['document'][:100]}...")

			print()


def rrf_search_results(query, k, enhance, limit, rerank_method, evaluate):

	print(f"original query - {query}")
	
	result = rrf_search_command(query, k, enhance, limit, rerank_method)
	
	if result["reranked"]:
		print(f"Re-ranking top {len(result['results'])} results using {result['rerank_method']} method...\n")
	
	if result["enhanced_query"]:
		print(f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n")
	
	print(f"Reciprocal Rank Fusion Search Results for '{result['query']}' (k = {result['k']}):")
	
	print()
	
	for i, res in enumerate(result["results"], 1):

		print(f"{i}. {res['title']}")
		
		if "Re-rank" in res:

			print(f"   Re-rank Score: {res.get('Re-rank', 0):.3f}/10")
		
		if result["rerank_method"] == "batch":
			print(f"   Re-rank Rank: {i}")
		
		if result["rerank_method"] == "cross_encoder":
			print(f"   Cross Encoder Score: {res['crossencoder_score']}")
			

		print(f"   RRF Score: {res.get('score', 0):.3f}")

		metadata = res.get("metadata", {})

		if "bm25_rank" in metadata and "semantic_rank" in metadata:

			print(f"   BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}")

			print(f"   {res['document'][:100]}...")

			print()
	
	return result








