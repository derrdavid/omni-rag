from openai import OpenAI
import os
from dotenv import load_dotenv
from vector_store import VectorStore

# Set your OpenAI API key here or use an environment variable
load_dotenv() 
vec = VectorStore(
    table_name="vectors",
    num_dimensions=1536,
    distance_type="cosine"
)
client = OpenAI()

def run_query(prompt: str, top_k: int = 1):
    search_list = vec.search(prompt, top_k=top_k)
    if not search_list:
        return "Kein Treffer gefunden."
    top_match = search_list[0]

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "developer",
                "content": f"Context: {top_match.contents}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.output_text

# Beispiel für direkten Aufruf in Jupyter:
# result = run_query("Wer ist Dirk?")
# print(result)
