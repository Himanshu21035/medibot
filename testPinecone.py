from google import genai
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

GOOGLE_API_KEY = "AIzaSyBMmOaSLStDSYA2PKgd-C1twoZDXsM_cAY"
PINECONE_API_KEY = "pcsk_7U6p86_DMVfbQKxRSWJ1XgjwppqCkbGcjrZaUBBuUsnc9XWqdbfnAQGUWa3dGB8nWKwWq3"

# Init embedding model (384 dimensions)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medical-chatbot"

# 1. Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,   # matches all-MiniLM-L6-v2 output
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# 2. Test text
text = "Insulin is a hormone that regulates blood sugar."

# 3. Create embedding
vector = model.encode(text).tolist()

# 4. Upsert into Pinecone
index.upsert([
    ("test-id-1", vector, {"text": text})
])

print("Vector inserted into Pinecone.")

# 5. Query the index
query_vector = model.encode(text).tolist()
query_resp = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print("\nQuery Results:")
for match in query_resp.matches:
    print(f"- Score: {match.score} | Text: {match.metadata['text']}")