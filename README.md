# Agentic RAG Bot

This project is an AI RAG (Retrieval-Augmented Generation) bot that uses a Streamlit interface to extract information from PDF files and answer user queries. The system is particularly optimized for retrieving and summarizing biographical information about people.

## ✨ Features

- **PDF Upload and Processing:** Upload PDF files and extract text content using `PDFPlumber`.
- **Vector Storage:** Store text chunks as vector representations in a `Qdrant` vector database.
- **Intelligent Query Processing:** Analyze queries using `Langchain` and `CrewAI` agents, determine the most appropriate information source (vector store or web search via `Tavily`), and generate responses.
- **Dynamic Response Generation:** Generate context-aware responses using `CrewAI` agents powered by local LLMs like Llama 3.1 via `Ollama`.
- **Web Interface:** An easy-to-use web interface built with `Streamlit` for interaction.
- **Data Validation:** Ensures data integrity and structure using `Pydantic`.

## 🚀 Setup

Follow these steps to set up and run this project on your local machine:

**1. Clone the Repository:**

```bash
git clone https://github.com/muhammedkurtulus/agentic-rag-chatbot.git
cd agentic-rag-bot
```

**2. Create and Activate a Virtual Environment (Recommended):**

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**4. Install Ollama:**

- Download and install Ollama for your operating system from the [Ollama website](https://ollama.com/).
- Download the required models:
  ```bash
  ollama pull llama3.1
  ollama pull bge-m3
  ```

**5. Set Up Environment Variables:**

- Create a file named `.env` in the project root.
- Add the necessary API keys and URLs to this file:
  ```dotenv
  QDRANT_URL="YOUR_QDRANT_INSTANCE_URL"
  QDRANT_API_KEY="YOUR_QDRANT_API_KEY" # Optional, depends on your Qdrant setup
  TAVILY_API_KEY="YOUR_TAVILY_API_KEY" # Required for web search
  OLLAMA_BASE_URL="http://localhost:11434" # This is usually the default
  ```
- Replace the placeholder values with your actual credentials.

## 💻 Usage

Once the setup is complete, you can start the Streamlit application with the following command:

```bash
streamlit run app.py
```

This command will open the application's interface in your web browser. You can now upload PDFs and chat with the bot.
