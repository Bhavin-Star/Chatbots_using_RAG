2) RAG Chatbot with Groq and LangChain

This project implements a hybrid Retrieval-Augmented Generation (RAG) chatbot.
The chatbot uses uploaded text data when it is relevant to the user’s query and otherwise responds using the language model’s general knowledge.

The goal is to keep responses natural, useful, and grounded without forcing irrelevant context into every answer.

Overview
Loads knowledge from a local text file (data.txt)
Converts the text into embeddings using a HuggingFace model
Stores embeddings in a Chroma vector database
Retrieves relevant chunks based on semantic similarity
Injects context only when similarity is high enough
Falls back to normal chatbot behavior when no relevant context is found
This prevents awkward responses for greetings or simple questions while still benefiting from RAG when appropriate.

Technologies Used
Python
Groq API (LLM backend)
LangChain
ChromaDB
HuggingFace Sentence Transformers

Project Structure
├── data.txt            # Source knowledge file
├── chroma_db/          # Persisted vector database
├── chatbot.py          # Main chatbot script
├── requirements.txt    # Dependencies
└── README.md           # Documentation

Setup Instructions
1. Clone the Repository
git clone <repository-url>
cd <repository-folder>

2. Create and Activate a Virtual Environment (Recommended)
python -m venv myenv

Windows
myenv\Scripts\activate

Linux / macOS
source myenv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

Environment Variable Configuration
The Groq API key is not stored in the code.
Set it as an environment variable:

Windows (CMD)
setx GROQ_API_KEY "your_groq_api_key"
Restart the terminal after running this.

Linux / macOS
export GROQ_API_KEY="your_groq_api_key"

Adding Your Data
Place your content inside:
data.txt
This file represents the knowledge base used by the chatbot.
You can update or replace it at any time.

Running the Chatbot
python chatbot.py

You should see:
RAG Chatbot ready (type 'exit' to quit)

How the System Works
Text is loaded from data.txt
The text is split into overlapping chunks
Each chunk is converted into an embedding
Embeddings are stored in a Chroma vector database
For each user query:
Similar documents are retrieved with similarity scores
Only documents below a similarity threshold are used as context
If relevant context exists, it is provided to the model
If no relevant context exists, the model responds normally
Why Similarity Filtering Is Used
Without filtering, irrelevant context can cause strange or confusing responses
(e.g., explaining technical topics when the user just says “hi”).

By filtering on similarity score:
Context is injected only when it makes sense
Greetings and simple questions behave naturally
RAG remains useful instead of intrusive

Limitations
No long-term conversation memory

Uses Llama 3.1 8B which was trained on data with a knowledge cutoff date of December 2023. 
The model itself was officially released by Meta on July 23, 2024. 
This means the model does not have information on events or developments that occurred after December 2023. 

Not intended for medical, legal, or professional advice

Security Notes
API keys are handled using environment variables
No secrets are committed to version control
Repository history should be reset if a key is ever exposed

Author:- Bhavin Shah
