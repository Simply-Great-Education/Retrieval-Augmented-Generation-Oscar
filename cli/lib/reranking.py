import os 
from dotenv import load_dotenv
from google import genai
import time
import json
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
	raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def rerank(query, docs, method, limit):
	match method:
		case "individual":
			return rerank_individual(query, docs, limit)
		
		case "batch":
			return rerank_batch(query, docs, limit)
		
		case "cross_encoder":
			return rerank_cross_encoder(query, docs, limit)
		
		case _:
			return query


def rerank_cross_encoder(query, docs, limit): # this function uses a cross encoder to get a relevance score from he query, and the document itself, then sorts them by ascending scores
	pairs = []
	for doc in docs:
		pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
	
	cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
	
	scores = cross_encoder.predict(pairs)
	
	for doc, score in zip(docs, scores):
		doc["crossencoder_score"] = float(score)
	
	sorted_result = sorted(docs, key=lambda x: x["crossencoder_score"], reverse=True)
	
	return sorted_result[:limit]
	


def rerank_batch(query, docs, limit):
	doc_map = {}
	doc_list = []
	for doc in docs:
		doc_id = doc["id"]
		doc_map[doc_id] = doc
		doc_list.append(f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}")

	doc_list_str = "\n".join(doc_list)


	prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
	
	response = client.models.generate_content(model='gemma-4-31b-it', contents=prompt)
	results = json.loads(response.text)
	
	reranked = []
	for i, doc_id in enumerate(results):
		if doc_id in doc_map:
			reranked.append({**doc_map[doc_id], "batch_rank": i + 1})
	
	return reranked[:limit]


def rerank_individual(query, docs, limit):
	results = []
	test = []
	
	for doc in docs:
		prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
		
		response = client.models.generate_content(model='gemma-4-31b-it', contents=prompt)
		score = response.text
		test = int(score)
		results.append({**doc, "Re-rank": test})
		time.sleep(5)
		
	
	sorted_result = sorted(results, key=lambda x: x["Re-rank"], reverse=True)
		
	return sorted_result[:limit]
		
