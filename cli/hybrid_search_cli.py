import argparse

from lib.hybrid_search import normalize_scores, weighted_search_results, rrf_search_results

def main() -> None:
	parser = argparse.ArgumentParser(description="Hybrid Search CLI")
	subparsers = parser.add_subparsers(dest="command", help="Available commands")
	
	normalize_parser = subparsers.add_parser("normalize", help="Normalize numbers using min-max normalization")
	normalize_parser.add_argument("nums", nargs="*", type=float, help="list of nums to be normalized")
	
	weighted_search_parser = subparsers.add_parser("weighted-search", help="weighted search")
	weighted_search_parser.add_argument("query", type=str, help="query for weighted search")
	weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="alpha thing")
	weighted_search_parser.add_argument("--limit", type=int, default=5, help="limit of results printed")
	
	rrf_search_parser = subparsers.add_parser("rrf-search", help="reciprocal Rank Fusion search")
	rrf_search_parser.add_argument("query", type=str, help="query for rrf search")
	rrf_search_parser.add_argument("-k", type=int, default=60, help="k parameter thing controls how much weight we give higher ranked results")
	rrf_search_parser.add_argument("--limit", type=int, default=5, help="num of results printed")
	rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
	rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch"], help="method used for re-ranking results")
	
	args = parser.parse_args()

	match args.command:
		case "normalize":
			results = normalize_scores(args.nums)
			for score in results:
				print(f"* {score:.4f}")
		
		case "weighted-search":
			weighted_search_results(args.query, args.alpha, args.limit)
		
		case "rrf-search":
			rrf_search_results(args.query, args.k, args.enhance, args.limit, args.rerank_method)
			
		case _:
			parser.print_help()

if __name__ == "__main__":
	main()