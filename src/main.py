from openai import OpenAI
import os
from dotenv import load_dotenv

# Set your OpenAI API key here or use an environment variable
load_dotenv() 

def main():
    # Hier können Sie Ihre RAG-Logik einfügen, z.B. Dokumente laden, Vektoren berechnen, etc.
    # rag_context = ...

    # Hier können Sie den Prompt ggf. mit RAG-Ergebnissen anreichern
    # prompt = augment_with_rag(prompt, rag_context)

    """ response = client.responses.create(
        model="gpt-4.1-nano",
        input="Nenne mir drei Vorteile von Retrieval-Augmented Generation."
    )

    print("Antwort von GPT-4.1-nano:")
    print(response.output_text) """

if __name__ == "__main__":
    main()
