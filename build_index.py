import os
import pickle
import mysql.connector
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("EMBED_MODEL")
INDEX_PATH = os.getenv("INDEX_PATH")
META_PATH = os.getenv("META_PATH")

DB = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

def fetch_menu_data():
    """Fetch all menu items from database"""
    try:
        conn = mysql.connector.connect(**DB)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, description, category, ingredients, price FROM menu")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except mysql.connector.Error as e:
        print(f"❌ Database error: {e}")
        return []

def create_search_texts(menu_items):
    """Create searchable text from menu items"""
    texts = []
    for item in menu_items:
        # Combine all relevant fields for searching
        search_text = f"{item['name']} {item['description']} {item['category']} {item['ingredients']}"
        texts.append(search_text)
    return texts

def main():
    print("🍽️  Building Menu Search Index...")
    
    # Validate environment variables
    required_vars = ["EMBED_MODEL", "INDEX_PATH", "META_PATH", "DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    print("📋 Fetching menu data from database...")
    menu_items = fetch_menu_data()
    
    if not menu_items:
        print("❌ No menu data found!")
        return
    
    print(f"✅ Loaded {len(menu_items)} menu items")
    
    # Create searchable texts
    texts = create_search_texts(menu_items)
    
    print(f"🤖 Loading embedding model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print("🔮 Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    
    print(f"📊 Embedding dimension: {embeddings.shape[1]}")
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(menu_items, f)
    
    print(f"💾 Index saved to: {INDEX_PATH}")
    print(f"💾 Metadata saved to: {META_PATH}")
    print("✅ Menu search index built successfully!")

if __name__ == "__main__":
    main()