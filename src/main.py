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

def main():
    prompt = "Wer ist Dirk?"
    search_list = vec.search(prompt, top_k=1)
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

    print("Antwort von GPT-4.1-nano:")
    print(response.output_text)

if __name__ == "__main__":
    main()
