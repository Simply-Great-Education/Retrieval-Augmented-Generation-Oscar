#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_text, semantic_chunk_text

def main():
	parser = argparse.ArgumentParser(description="Semantic Search CLI")
	subparsers = parser.add_subparsers(dest="command", help="Available commands")	
	
	verify_parser = subparsers.add_parser("verify", help="Verify module")
	
	embed_parser = subparsers.add_parser("embed_text", help="Embed text")
	embed_parser.add_argument("text", type=str, help="text to be embeded")
	
	verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")
	
	embed_query_parser = subparsers.add_parser("embedquery", help="Embed query")
	embed_query_parser.add_argument("query", type=str, help="query to be embeded")
	
	search_parser = subparsers.add_parser("search", help="search for query")
	search_parser.add_argument("query", type=str, help="query to be searched for")
	search_parser.add_argument("--limit", type=int, default=5, help="Optional limit")
	
	chunk_parser = subparsers.add_parser("chunk", help="chunk text")
	chunk_parser.add_argument("text", type=str, help="query to be chunked")
	chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Optional chunk limit")
	chunk_parser.add_argument("--overlap", type=int, default=0, help="word overlap of chunks")
	
	semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="chunk text using semantics")
	semantic_chunk_parser.add_argument("text", type=str, help="query to be chunked")
	semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="max chunk size limit")
	semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="word overlap of chunks")
	
	args = parser.parse_args()

	match args.command:
		case "verify":
			verify_model()
		
		case "embed_text":
			embed_text(args.text)
		
		case "verify_embeddings":
			verify_embeddings()
		
		case "embedquery":
			embed_query_text(args.query)
		
		case "search":
			search_command(args.query, args.limit)
		
		case "chunk":
			chunk_text(args.text, args.chunk_size, args.overlap)
		
		case "semantic_chunk":
			semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
			
		case _:
			parser.print_help()

if __name__ == "__main__":
	main()