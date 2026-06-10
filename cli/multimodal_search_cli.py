import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command", help="Available commands")
	
	verify_parser = subparsers.add_parser("verify_image_embedding")
	verify_parser.add_argument("image", type=str, help="the path to an image file")
	
	img_search_parser = subparsers.add_parser("image_search")
	img_search_parser.add_argument("image", type=str, help="the path to an image file")
	
	args = parser.parse_args()
	
	match args.command:
		case "verify_image_embedding":
			verify_image_embedding(args.image)
		
		case "image_search":
			results = image_search_command(args.image)
			
			for i, res in enumerate(results):
				print(f"{i + 1}. {res['title']} (similarity: {res['score']})")
				print(f"	{res['description'][:100]}...")
				print()
		
		case _:
			parser.print_help()
	
	
if __name__ == "__main__":
	main()