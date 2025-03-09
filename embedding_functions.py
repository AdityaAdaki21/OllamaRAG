import requests
from typing import List
from chromadb.utils import embedding_functions

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function for ChromaDB that uses Ollama."""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        results = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text,
                    "keep_alive": "5m"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Error getting embeddings: {response.text}")
            
            results.append(response.json()["embedding"])
        
        return results