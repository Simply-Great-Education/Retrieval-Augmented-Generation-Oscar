import os
from dotenv import load_dotenv
from google import genai
from time import sleep

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
	raise RuntimeError("GEMINI_API_KEY environment variable not set")
	
client = genai.Client(api_key=api_key)

model = "gemma-4-31b-it"
# model = "models/gemma-4-26b-a4b-it"


# for m in client.models.list():

    #print(m.name, "-", m.supported_actions)
    
prompt = "Pick a number 1 to 100, only return your choice in number and nothing else"
    
for i in range(0, 3):
	print(i)
	response = client.models.generate_content(model=model, contents=prompt)
	print(response.text)
	sleep(3)

			
prompt = f"Rate how well this movie matches the search query. Query: {query} Movie: {doc.get('title', "")}Consider: Direct relevance to query, User intent (what they're looking for), Content appropriateness. Rate 0-10 (10 = perfect match). Output ONLY the number in your response"


