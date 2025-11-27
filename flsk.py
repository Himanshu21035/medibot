import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
# import google.generativeai as genai
from google import genai

app = Flask(__name__)
CORS(app)

# Configuration
GOOGLE_API_KEY = "AIzaSyBMmOaSLStDSYA2PKgd-C1twoZDXsM_cAY"
PINECONE_API_KEY ="pcsk_7U6p86_DMVfbQKxRSWJ1XgjwppqCkbGcjrZaUBBuUsnc9XWqdbfnAQGUWa3dGB8nWKwWq3"
INDEX_NAME = "medical-chatbot"

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# genai.configure(api_key=GOOGLE_API_KEY)
# gemini_model = genai.GenerativeModel("gemini-pro")
client = genai.Client(api_key="AIzaSyBMmOaSLStDSYA2PKgd-C1twoZDXsM_cAY")


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


@app.route('/ingest', methods=['POST'])
def ingest_documents():
    """
    Ingest documents into Pinecone vector database.
    Expected JSON: {
        "documents": [
            {"id": "doc1", "text": "content here"},
            {"id": "doc2", "text": "more content"}
        ]
    }
    """
    try:
        data = request.json
        documents = data.get("documents", [])
        
        if not documents:
            return jsonify({"error": "No documents provided"}), 400
        
        # Create embeddings and prepare for upsert
        vectors = []
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text")
            
            if not doc_id or not text:
                continue
            
            # Generate embedding
            embedding = embedding_model.encode(text).tolist()
            
            # Prepare vector with metadata
            vectors.append((doc_id, embedding, {"text": text}))
        
        # Upsert to Pinecone
        index.upsert(vectors)
        
        return jsonify({
            "message": f"Successfully ingested {len(vectors)} documents",
            "count": len(vectors)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    RAG endpoint: Retrieve relevant context and generate answer.
    Expected JSON: {
        "question": "What is insulin?",
        "top_k": 3  # optional, default 3
    }
    """
    try:
        data = request.json
        question = data.get("question", "")
        top_k = data.get("top_k", 3)
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Step 1: Generate query embedding
        query_embedding = embedding_model.encode(question).tolist()
        
        # Step 2: Retrieve relevant documents from Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Step 3: Extract context from search results
        context_chunks = []
        for match in search_results.matches:
            if match.score > 0.5:  # relevance threshold
                context_chunks.append(match.metadata.get('text', ''))
        
        if not context_chunks:
            return jsonify({
                "answer": "I don't have enough relevant information to answer this question.",
                "sources": []
            }), 200
        
        # Step 4: Create augmented prompt
        context = "\n\n".join(context_chunks)
        augmented_prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}

Using the CONTEXT provided, answer the QUESTION accurately and concisely. 
Keep your answer grounded in the facts of the CONTEXT. 
If the CONTEXT doesn't contain the answer to the QUESTION, answer it with your best knowledge.
"""
        
        # Step 5: Generate answer using Gemini
        response = client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=augmented_prompt
                )
        # Step 6: Return answer with sources
        sources = [
            {
                "text": match.metadata.get('text', '')[:200] + "...",
                "score": float(match.score),
                "id": match.id
            }
            for match in search_results.matches if match.score > 0.5
        ]
        
        return jsonify({
            "answer": response.text,
            "sources": sources,
            "retrieved_count": len(sources)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "index_name": INDEX_NAME,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gemini-pro"
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
