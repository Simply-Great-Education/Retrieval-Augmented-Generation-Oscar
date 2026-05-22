import os 
from dotenv import load_dotenv
from google import genai
import time


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
	raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def rerank(query, docs, method, limit):
	match method:
		case "individual":
			return rerank_individual(query, docs, limit)
		
		case _:
			return query


def rerank_individual(query, docs, limit):
	results = []
	
	print("^>^")
	
	for doc in docs:
		prompt = f"""
			Rate how well this movie matches the search query.

			Query: "{query}"
			Movie: {doc.get("title", "")}

			Consider:
			- Direct relevance to query
			- User intent (what they're looking for)
			- Content appropriateness

			Rate 0-10 (10 = perfect match).
			Output ONLY the number in your response, no other text or explanation.

			Score:
			"""
		
		response = client.models.generate_content(model='gemma-4-31b-it', contents=prompt)
		score = response.text
		test = int(score)
		results.append({**doc, "Re-rank": test})
		time.sleep(5)
	
	sorted_result = sorted(results, key=lambda x: x["Re-rank"], reverse=True)
		
	return sorted_result[:limit]
		
