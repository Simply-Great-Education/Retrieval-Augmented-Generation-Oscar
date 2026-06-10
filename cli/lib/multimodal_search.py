from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies


class MultimodalSearch:
	def __init__(self, docs="", model_name="clip-ViT-B-32"):
		self.model = SentenceTransformer(model_name)
		self.documents = docs
		self.texts = []
		for doc in docs:
			self.texts.append(f"{doc['title']}: {doc['description']}")
		self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)# creating embeddings of all the docs (in this case all the items in the movies database)
	
	
	def embed_image(self, image_path): # opening and encoding the specified image
		image = Image.open(image_path)
		return self.model.encode([image])[0]
	
	
	def search_with_image(self, image_path):
		image_embedding = self.embed_image(image_path) # embedding the image to be searched with
		results = []
		for i, text_embedding in enumerate(self.text_embeddings): # looping through all embeded items and calculating the cosine similarity between each item and the image embedding
			cosine = cosine_similarity(image_embedding, text_embedding)
			results.append({
				"doc_id": self.documents[i]['id'],
				"title": self.documents[i]['title'],
				"description": self.documents[i]['description'],
				"score": cosine
				})
		
		sorted_result = sorted(results, key=lambda x: x["score"], reverse=True) # sorting results by their cosine similarity (score)
		
		return sorted_result[:5]
			



def verify_image_embedding(image_path):
	MS = MultimodalSearch()
	embedding = MS.embed_image(image_path)
	print(f"Embedding shape: {embedding.shape[0]} dimensions")
	

def image_search_command(image_path):
	movies = load_movies()
	MS = MultimodalSearch(movies)
	return MS.search_with_image(image_path)


