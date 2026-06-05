import argparse

from lib.augmented_generation import rag_command, sum_command, citation_command, question_command

def main() -> None:
	parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
	subparsers = parser.add_subparsers(dest="command", help="Available commands")

	rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
	rag_parser.add_argument("query", type=str, help="Search query for RAG")
	
	sum_parser = subparsers.add_parser("summarize", help="Summarize multiple docs or smt")
	sum_parser.add_argument("query", type=str, help="Search query for sum")
	sum_parser.add_argument("--limit", type=int, default=5, help="Limit for search")
	
	citation_parser = subparsers.add_parser("citations", help="Make citations for search results")
	citation_parser.add_argument("query", type=str, help="Search query")
	citation_parser.add_argument("--limit", type=int, default=5, help="Limit for search")
	
	question_parser = subparsers.add_parser("question", help="Question for llm search")
	question_parser.add_argument("question", type=str, help="Search question")
	question_parser.add_argument("--limit", type=int, default=5, help="Limit for search")

	args = parser.parse_args()

	match args.command:
		case "rag":
			query = args.query
			result = rag_command(query)
			print("Search Results:") # printing results from rrf search and also from RAG agent (llm)
			for doc in result['search_results']:
				print(f"- {doc['title']}")
			print()
			print("RAG Response:")
			print(result['answer'])
			
		case "summarize":
			result = sum_command(args.query, args.limit)
			print("Search Results:") # printing results from rrf search and also from llm summary
			for doc in result['search_results']:
				print(f"- {doc['title']}")
			print()
			print("LLM Summary:")
			print(result['summary'])
			
		case "citations":
			result = citation_command(args.query, args.limit)
			print("Search Results:") # printing results from rrf search and also from llm citations
			for doc in result['search_results']:
				print(f"- {doc['title']}")
			print()
			print("LLM Answer:")
			print(result['citations'])
			
		case "question":
			result = question_command(args.question, args.limit)
			print("Search Results:") # printing results from rrf search and also from llm question
			for doc in result['search_results']:
				print(f"- {doc['title']}")
			print()
			print("Answer:")
			print(result['answer'])
			
		case _:
			parser.print_help()

if __name__ == "__main__":
	main()