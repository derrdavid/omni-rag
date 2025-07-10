from openai import OpenAI
from dotenv import load_dotenv
from vector_store import VectorStore
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

class DecomposedOutput(BaseModel):
    decomposed: list[str]

# Set your OpenAI API key here or use an environment variable
load_dotenv() 
vec = VectorStore(
    table_name="vectors",
    num_dimensions=1536,
    distance_type="cosine"
)
client = OpenAI()
reranker_model = SentenceTransformer("BAAI/bge-base-en-v1.5")      

def rewrite_query(query):
    prompt = "Rewrite the user query for retrieval."
    result = client.responses.create(
        model="gpt-4.1-nano",
        input=[{"role": "developer", "content": prompt}, {"role": "user", "content": query}]
    )
    return result.output_text

def decompose_query(query):
    prompt = "Decompose the query into clear sub-queries."
    result = client.responses.parse(
        model="gpt-4.1-nano",
        input=[{"role": "developer", "content": prompt}, {"role": "user", "content": query}],
        text_format=DecomposedOutput
    )
    return result.output_parsed.decomposed

def retrieve(sub_queries, top_k=3):
    docs, seen = [], set()
    for q in sub_queries:
        for d in vec.search(q, top_k=top_k) or []:
            doc_id = getattr(d, 'id', str(d))
            if doc_id not in seen:
                docs.append(d)
                seen.add(doc_id)
    return docs

def rerank(query, docs, top_n=3):
    q_emb = reranker_model.encode(query, convert_to_tensor=True)
    d_emb = reranker_model.encode([d.contents for d in docs], convert_to_tensor=True)
    sims = reranker_model.similarity(q_emb, d_emb)
    idx = sims.argsort(descending=True)
    return [docs[i] for i in idx[0][:top_n].tolist()]

def generate(query, docs):
    context = "\n".join(d.contents for d in docs)
    result = client.responses.create(
        model="gpt-4.1-nano",
        input=[{"role": "developer", "content": f"Context: {context}"}, {"role": "user", "content": query}]
    )
    return result.output_text

def omni_rag(query):
    q_rewritten = rewrite_query(query)
    sub_queries = decompose_query(q_rewritten)
    docs = retrieve(sub_queries)
    if not docs:
        return "No relevant documents found."
    top_docs = rerank(query, docs)
    return generate(query, top_docs)

if __name__ == "__main__":
    query = input("Query: ")
    # query = "Was passierte in New Mexico, Albuquerue in einer Kirche?"
    print(omni_rag(query))
