from openai import OpenAI
import os
from dotenv import load_dotenv
from vector_store import VectorStore
from transformers import pipeline

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
    
    print(top_match)
    
    prompt = "Um was ging der Beef zwischen Elon Musk und Trump?"
    
    pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True)
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
    pipe(input)

# Beispiel für direkten Aufruf in Jupyter:
# result = run_query("Wer ist Dirk?")
# print(result)
