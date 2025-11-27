from google import genai

client = genai.Client(api_key="AIzaSyBMmOaSLStDSYA2PKgd-C1twoZDXsM_cAY")
text = "Diabetes is a chronic disease that affects how your body turns food into energy."

# Use the embedding model
resp = client.models.embed_content(
    model="models/text-embedding-004",  # Best universal embedding model
    contents=text
)

embedding = resp.embeddings[0].values
print(f"Embedding length: {len(embedding)}")
print(embedding[:10])  # Print first 10 values
