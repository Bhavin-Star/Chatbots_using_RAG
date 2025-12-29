import os
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=api_key)


loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
splits = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})


print("\nRAG Chatbot ready (type 'exit' to quit)\n")

SIMILARITY_THRESHOLD = 0.7  # lower = more similar

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye.")
        break

    # Retrieve documents WITH scores
    results = db.similarity_search_with_score(user_input, k=3)

    # Keep only relevant docs
    relevant_docs = [
        doc.page_content for doc, score in results if score < SIMILARITY_THRESHOLD
    ]

    context = "\n\n".join(relevant_docs)

    if context:
        prompt = f"""
You are a helpful assistant.
Use the following context only if it is relevant to the user's question.

Context:
{context}

User:
{user_input}
"""
    else:
        prompt = f"""
You are a helpful assistant.

User:
{user_input}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nBot:", response.choices[0].message.content, "\n")
