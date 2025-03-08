import chromadb
import os

# Ensure directory exists
os.makedirs("fresh_chroma_db", exist_ok=True)

# Create a fresh client
client = chromadb.PersistentClient(path="fresh_chroma_db")

# Try to create a collection
try:
    collection = client.create_collection(name="test_collection")
    print("Successfully created collection!")
    
    # Try adding some data
    collection.add(
        ids=["id1", "id2"],
        documents=["This is a test", "This is another test"],
        metadatas=[{"source": "test1"}, {"source": "test2"}]
    )
    print("Successfully added data!")
    
    # Try retrieving data
    results = collection.query(
        query_texts=["test"],
        n_results=2
    )
    print("Retrieved data:", results)
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")