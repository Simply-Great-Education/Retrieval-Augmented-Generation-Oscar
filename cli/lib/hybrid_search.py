import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import format_search_result, load_movies

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
		print(f"docmap size: {len(self.idx.docmap)}", flush=True)
		return self.idx.bm25_search(query, limit)

	def weighted_search(self, query, alpha, limit=5):
		print("calling bm25...", flush=True)
		bm25_results = self._bm25_search(query, (limit * 500))
		print(f"got {len(bm25_results)} bm25 results", flush=True)
		print("calling semantic...", flush=True)
		semantic_results = self.semantic_search.search_chunks(query, (limit * 500))
		print(f"got {len(semantic_results)} semantic results", flush=True)
		
		print(f"bm25: {len(bm25_results)}, semantic: {len(semantic_results)}", flush=True)
		
		if bm25_results:
			print(f"bm25 sample keys: {list(bm25_results[0].keys())}", flush=True)
		if semantic_results:
			print(f"semantic sample keys: {list(semantic_results[0].keys())}", flush=True)
		
		scores = [r["score"] for r in bm25_results]
		normalized_bm25 = normalize_scores(scores)
		
		scores = [r["score"] for r in semantic_results]
		normalized_semantic = normalize_scores(scores)
		print("combining...", flush=True)
		combined = combine_search_results(normalized_bm25, normalized_semantic, alpha)
		print(f"combined: {len(combined)}", flush=True)
		return combined[:limit]

	def rrf_search(self, query, k, limit=10):
		raise NotImplementedError("RRF hybrid search is not implemented yet.")
	

def normalize_scores(scores):
	if scores == None:
		return
		
	small = min(scores)
	large = max(scores)
	
	if small == large:
		for score in scores:
			print(f"* 1.0")
		return
	
	results = []
	for num in scores:
		score = (num - small) / (large - small)
		scores.append(score)
	
	return results

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
	return alpha * bm25_score + (1 - alpha) * semantic_score
	


def combine_search_results(bm25_scores, semantic_scores, alpha):

	combined_scores = {}
	
	print("combining: normalizing bm25...", flush=True)

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
			combined_scores[doc_id]["bm25_score"] = result["normalized_score"]
	
	print("combining: normalizing semantic...", flush=True)

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
			combined_scores[doc_id]["semantic_score"] = result["normalized_score"]
	
	print(f"combined dict size: {len(combined_scores)}", flush=True)

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



def weighted_search_results(query, alpha, limit):
	
	result = weighted_search_command(query, alpha, limit)
	
	print(f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):")
	print(f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic")
	
	for i, res in enumerate(result["results"], 1):
		print(f"{i}. {res['title']}")
		print(f"   Hybrid Score: {res.get('score', 0):.3f}")
		metadata = res.get("metadata", {})
		if "bm25_score" in metadata and "semantic_score" in metadata:
			print(f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}")
			print(f"   {res['document'][:100]}...")
			print()











