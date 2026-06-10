import argparse
import os
import mimetypes

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
	raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--image", required=True, type=str, help="the path to an image file")
	parser.add_argument("--query", required=True, type=str, help="a text query to rewrite based on the image")
	
	args = parser.parse_args()
	
	mime, _ = mimetypes.guess_type(args.image)
	print(f"file - {args.image} mime - {mime}")
	mime = mime or "image/jpeg"
	
	with open(args.image, 'rb') as f:
		data = f.read()
	
	system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
	
	parts = [
		system_prompt,
		types.Part.from_bytes(data=data, mime_type=mime),
		args.query.strip(),
	]
	
	response = client.models.generate_content(model='gemma-4-31b-it', contents=parts)
	
	print(f"Rewritten query: {response.text.strip()}")
	if response.usage_metadata is not None:
		print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
	main()