import ollama
import numpy as np

# Models
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

VECTOR_DB = []


def load_dataset(file_path):
    """
    Load text file and return chunks.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = file.readlines()

    print(f"Loaded {len(chunks)} entries")
    return chunks


def get_embedding(text):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response["embeddings"][0]


def build_vector_db(dataset):
    """
    Create embeddings for all chunks.
    """
    global VECTOR_DB

    VECTOR_DB = []

    for i, chunk in enumerate(dataset):
        embedding = get_embedding(chunk)

        VECTOR_DB.append(
            {
                "text": chunk.strip(),
                "embedding": embedding
            }
        )

        print(f"Added chunk {i+1}/{len(dataset)}")

    return VECTOR_DB


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(query, top_n=3):
    """
    Retrieve most relevant chunks.
    """

    query_embedding = get_embedding(query)

    similarities = []

    for item in VECTOR_DB:
        similarity = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        similarities.append(
            (
                item["text"],
                similarity
            )
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_n]


def generate_response(query, top_n=3):
    """
    Full RAG pipeline.
    """

    retrieved_docs = retrieve(query, top_n)

    context = "\n".join(
        [doc for doc, _ in retrieved_docs]
    )

    system_prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:
"I couldn't find that information in the provided documents."

Context:
{context}
"""

    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    return (
        response["message"]["content"],
        retrieved_docs
    )


if __name__ == "__main__":

    dataset = load_dataset("cat-facts.txt")

    build_vector_db(dataset)

    while True:
        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            break

        answer, docs = generate_response(query)

        print("\nRetrieved Documents:\n")

        for doc, score in docs:
            print(f"({score:.4f}) {doc}")

        print("\nAnswer:\n")
        print(answer)