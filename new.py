from google import genai

client = genai.Client(api_key="AIzaSyBMmOaSLStDSYA2PKgd-C1twoZDXsM_cAY")
response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="Explain what diabetes is in one sentence."
)

print("\nLLM Response:\n")
print(response.text)
