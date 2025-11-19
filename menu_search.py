import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("EMBED_MODEL")
INDEX_PATH = os.getenv("INDEX_PATH")
META_PATH = os.getenv("META_PATH")

class MenuSearch:
    def __init__(self):
        self.model = None
        self.index = None
        self.menu_items = None
        self.load_resources()
    
    def load_resources(self):
        """Load the model, index, and metadata"""
        print("🔄 Loading search resources...")
        
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            print("❌ Index files not found. Please run build_index.py first.")
            return False
        
        try:
            self.model = SentenceTransformer(MODEL_NAME)
            self.index = faiss.read_index(INDEX_PATH)
            
            with open(META_PATH, "rb") as f:
                self.menu_items = pickle.load(f)
            
            print(f"✅ Loaded {len(self.menu_items)} menu items")
            return True
        except Exception as e:
            print(f"❌ Error loading resources: {e}")
            return False
    
    def search(self, query, top_k=5):
        """Search for menu items similar to the query"""
        if self.model is None or self.index is None:
            print("❌ Search system not properly initialized")
            return []
        
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.menu_items):
                menu_item = self.menu_items[idx]
                results.append({
                    'rank': i + 1,
                    'score': float(1 / (1 + distance)),  # Convert distance to similarity score
                    'id': menu_item['id'],
                    'name': menu_item['name'],
                    'description': menu_item['description'],
                    'category': menu_item['category'],
                    'ingredients': menu_item['ingredients'],
                    'price': float(menu_item['price'])
                })
        
        return results
    
    def format_results(self, results):
        """Format search results for display"""
        if not results:
            return "No results found."
        
        output = []
        for result in results:
            output.append(f"""
🍽️  {result['name']} (Score: {result['score']:.3f})
📝 {result['description']}
🏷️  Category: {result['category']}
🛒 Price: ${result['price']:.2f}
📋 Ingredients: {result['ingredients']}
{'='*50}""")
        
        return "\n".join(output)

def main():
    # Initialize search system
    search_system = MenuSearch()
    
    if not search_system.load_resources():
        return
    
    print("\n🔍 Menu Search System Ready!")
    print("Type your search query or 'quit' to exit\n")
    
    while True:
        try:
            query = input("🍴 What are you looking for? ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\n🔎 Searching for: '{query}'")
            results = search_system.search(query, top_k=3)
            
            if results:
                print(f"\n✅ Found {len(results)} results:")
                print(search_system.format_results(results))
            else:
                print("❌ No matching menu items found.")
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during search: {e}")

if __name__ == "__main__":
    main()