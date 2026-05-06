import argparse

from lib.hybrid_search import normalize_scores, weighted_search_results

def main() -> None:
	parser = argparse.ArgumentParser(description="Hybrid Search CLI")
	subparsers = parser.add_subparsers(dest="command", help="Available commands")
	
	normalize_parser = subparsers.add_parser("normalize", help="Normalize numbers using min-max normalization")
	normalize_parser.add_argument("nums", nargs="*", type=float, help="list of nums to be normalized")
	
	weighted_search_parser = subparsers.add_parser("weighted-search", help="weighted search")
	weighted_search_parser.add_argument("query", type=str, help="query for weighted search")
	weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="query for weighted search")
	weighted_search_parser.add_argument("--limit", type=int, default=5, help="query for weighted search")
	
	args = parser.parse_args()

	match args.command:
		case "normalize":
			normalize_scores(args.nums)
		
		case "weighted-search":
			weighted_search_results(args.query, args.alpha, args.limit)
			
		case _:
			parser.print_help()

if __name__ == "__main__":
	main()