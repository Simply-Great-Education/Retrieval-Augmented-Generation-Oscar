from .search_utils import load_movies
from .hybrid_search import HybridSearch
from .semantic_search import SemanticSearch

from dotenv import load_dotenv
from google import genai
import json
import os

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
	raise RuntimeError("GEMINI_API_KEY environment variable not set")
	
client = genai.Client(api_key=api_key)

def rag_command(query, limit=5):
	movies = load_movies() # loading the movie database then initialising the hybridsearch class.
	hybrid_search = HybridSearch(movies)
	
	search_results = hybrid_search.rrf_search(query, k=60, limit=5) # performing an rrf search using the given query
	
	answer = generate_answer(query, search_results, limit=5) # calling generate_answer function to get response from llm
	
	return {
		"query": query,
		"search_results": search_results[:limit], # returning dict with data
		"answer": answer
	}

def sum_command(query, limit):
	movies = load_movies() # loading the movie database then initialising the hybridsearch class.
	hybrid_search = HybridSearch(movies)
	search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
	answer = generate_sum(query, search_results, limit) # calling sum function and getting result
	
	return {
		"query": query,
		"search_results": search_results[:limit], # returning dict with data
		"summary": answer
	}

def citation_command(query, limit):
	movies = load_movies() # loading the movie database then initialising the hybridsearch class.
	hybrid_search = HybridSearch(movies)
	search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
	answer = generate_citations(query, search_results, limit)
	
	return {
		"query": query,
		"search_results": search_results[:limit], # returning dict with data
		"citations": answer
	}

def question_command(question, limit):
	movies = load_movies() # loading the movie database then initialising the hybridsearch class.
	hybrid_search = HybridSearch(movies)
	search_results = hybrid_search.rrf_search(question, k=60, limit=limit)
	answer = generate_question(question, search_results, limit) # calling question function and getting result
	
	return {
		"question": question,
		"search_results": search_results[:limit], # returning dict with data
		"answer": answer
	}

	
	
def generate_answer(query, search_results, limit):
	retrieved_docs = []
	for result in search_results: # creating a list of all doc titles from the search results.
		retrieved_docs.append(f"{result['title']}, {result['document']}")
	
	
	prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.
Query: {query}
Documents: {retrieved_docs}
Answer:"""
	
	response = client.models.generate_content(model='gemma-4-31b-it', contents=prompt) # calling an llm with the prompt written above^
	
	return response.text

def generate_sum(query, search_results, limit):
	prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search results: {search_results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

	response = client.models.generate_content(model='gemma-4-31b-it', contents=prompt) # calling an llm with the prompt written above^
	
	return response.text

def generate_citations(query, search_results, limit):
	prompt = f"""Answer the query below and give information based on the provided documents. The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.
Query:{query}
Documents:{search_results}
Instructions:
Provide a comprehensive answer that addresses the query and cite sources in the format [1], [2], etc. when referencing information. If sources disagree, mention the different viewpoints, If the answer isn't in the provided documents, say "I don't have enough information" and be direct and informative
Answer:"""

	response = client.models.generate_content(model='gemma-4-31b-it', contents=prompt) # calling an llm with the prompt written above^
	
	return response.text


def generate_question(question, search_results, limit):
	retrieved_docs = []
	for result in search_results: # creating a list of all doc titles from the search results.
		retrieved_docs.append(f"{result['title']}, {result['document']}")
	
	prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.
Question: {question}
Documents: {retrieved_docs}
Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation
Answer:"""

	response = client.models.generate_content(model='gemma-4-31b-it', contents=prompt) # calling an llm with the prompt written above^
	
	return response.text

