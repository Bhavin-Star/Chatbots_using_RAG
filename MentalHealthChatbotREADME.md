1. Mental Health RAG Chatbot

This project is a mental health support chatbot built using a Retrieval-Augmented Generation (RAG) approach.
The chatbot answers user queries based only on predefined intent data to avoid hallucinations and unsafe responses.

Overview:
The chatbot loads intent data from a JSON file.
Relevant intents are retrieved using semantic search.
Responses are generated using a Groq-hosted language model.
The system is designed to be supportive, safe, and non-clinical.
Technologies Used
Python
Groq API (LLM)
LangChain
ChromaDB
HuggingFace Sentence Transformers

Project Structure
├── MentalHealthChatbot_RAG.py   # Main application file
├── intents.json                # Intent patterns and responses
├── intents_db/                 # Chroma vector database
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation

Setup Instructions
1. Clone the Repository
git clone <repository-url>
cd <repository-folder>

2. Create and Activate Virtual Environment
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

Running the Application
python MentalHealthChatbot_RAG.py


Expected output:
Mental Health RAG Bot is ready (type 'exit' to quit)

How It Works:
Intent data is loaded from intents.json
Intent text is converted into document chunks
Embeddings are generated using a HuggingFace model
Embeddings are stored in a Chroma vector database
Relevant intent chunks are retrieved for each user query
The retrieved context is sent to the Groq LLM
The model generates a response using only the retrieved context
Security Notes
API keys are handled using environment variables
No secrets are committed to the repository
Repository history should be reset if a key is ever exposed

Limitations:
Responses are limited to predefined intent data
No long-term conversation memory
Not intended for medical diagnosis or emergency situations

Disclaimer
This chatbot is intended for educational and supportive purposes only.
It does not replace professional mental health care.

Author:- Bhavin Shah
