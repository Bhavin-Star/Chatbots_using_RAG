import os
import json
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Groq client (API key from global env variable)
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable not set")

#print(os.getenv("GROQ_API_KEY"))

# 2. Load intents.json
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

intents_list = data["intents"]

# 3. Convert intents to documents (RAG)
documents = []

for intent in intents_list:
    text = (
        f"Intent: {intent.get('tag')}\n"
        f"Patterns: {', '.join(intent.get('patterns', []))}\n"
        f"Responses: {', '.join(intent.get('responses', []))}"
    )
    documents.append(Document(page_content=text))

# 4. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = splitter.split_documents(documents)

# 5. Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./intents_db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# 6. Chat loop (RAG)
print("\nMental Health RAG Bot is ready 💙 (type 'exit' to quit)\n")

while True:
    ask = input("You: ")
    if ask.lower() in ["exit", "quit"]:
        break

    # 🔍 Retrieve relevant intents
    retrieved_docs = retriever.invoke(ask)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are a helpful, empathetic mental health support assistant.
Use ONLY the information provided in the context.
If the user needs help beyond the context, respond gently and supportively.
Do NOT hallucinate.

Context:
{context}

User:
{ask}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    print("\nBot:", response.choices[0].message.content, "\n")
