import os, pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_NAME = os.getenv("EMBED_MODEL")
INDEX_PATH = os.getenv("INDEX_PATH")
META_PATH = os.getenv("META_PATH")

print("📌 Loading model, index, metadata...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data["query"]

    emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(emb, 5)

    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
