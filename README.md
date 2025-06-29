# RAG Assistant

Welcome to **RAG Assistant**, an intelligent assistant powered by **Retrieval-Augmented Generation (RAG)** to provide precise technical answers from PDF documents and web sources. Built with **FastAPI**, **LangChain**, and **Ollama**, this open-source project uses the **Mistral** model (or lighter alternatives like `phi3`) for efficient, multi-language responses (primarily French). It combines vector search (FAISS) and keyword search (BM25) for accurate document retrieval.

## 🚀 Key Features

- **RAG Pipeline**: Combines FAISS vector search and BM25 for accurate, context-aware answers.
- **Data Sources**: Supports PDFs (`app/data/pdfs/`) and web URLs (via `.env`).
- **Multi-Language Support**: Optimized for French, with potential for other languages.
- **Modern API**: FastAPI with Swagger UI for easy interaction.
- **Modular Design**: Extensible for adding new data sources or models.
- **Open-Source**: Uses free, open-source tools (LangChain, Ollama, FAISS).

## 📂 Project Structure

```
ecommerce-rag-assistant/
├── .env                    # Environment variables
├── app/
│   ├── api/               # FastAPI endpoints
│   │   ├── main.py        # API entry point
│   │   └── routes/rag.py  # RAG endpoint
│   ├── chains/            # RAG chain configuration
│   │   └── rag_chain.py
│   ├── core/              # Configuration and logging
│   │   ├── config.py
│   │   └── logger.py
│   ├── data/              # Data storage
│   │   └── pdfs/          # PDF documents
│   ├── ingestion/         # Data loading and preprocessing
│   │   ├── pdf_loader.py
│   │   ├── splitter.py
│   │   └── web_loader.py
│   ├── models/            # LLM configuration
│   │   └── llm_factory.py
│   ├── services/          # RAG pipeline logic
│   │   └── rag_service.py
│   └── vectorstore/       # Vector index management
│       └── retriever.py
├── create_index.py        # Script to build vector index
├── requirements.txt       # Dependencies
├── tests/                 # Unit tests
│   └── test_rag.py
└── vectorstore/           # FAISS index storage
```

## 🛠️ Prerequisites

- **Python**: 3.8 or higher
- **Ollama**: For running the LLM locally (Mistral, phi3, or tinyllama)
- **System**:
  - **RAM**: Minimum 8 GB (16 GB recommended for `mistral`; 4-8 GB sufficient for `phi3` or `tinyllama`)
  - **OS**: Windows, Linux (Ubuntu tested), or macOS
  - **CPU**: Multi-core recommended
  - **GPU**: Optional (NVIDIA with CUDA for better performance)
- **Disk Space**: ~1 GB for models and vector index
- **Dependencies**: Listed in `requirements.txt`

## 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/adamaKomi/rag.git
   cd rag
   ```

2. **Create a virtual environment**:

   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Linux/macOS**:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**:
   - Download and install Ollama from [ollama.com](https://ollama.com/download).
   - Pull a lightweight model (recommended for low-memory systems):
     ```bash
     ollama pull phi3
     ```
     Or use `mistral` if you have sufficient RAM (>16 GB):
     ```bash
     ollama pull mistral
     ```

5. **Set up environment variables**:
   - Create a `.env` file at the project root:
     ```
     PDF_DIR=app/data/pdfs/
     VECTORSTORE_DIR=vectorstore/
     USER_AGENT=MyApp/1.0
     WEB_SOURCES=https://example.com,https://another-example.com  # Optional
     ```

6. **Prepare PDF documents**:
   - Create the directory `app/data/pdfs/` if it doesn’t exist:
     ```bash
     mkdir -p app/data/pdfs
     ```
   - Place PDF files in `app/data/pdfs/`.

7. **Create the vector index**:

   ```bash
   python create_index.py
   ```

8. **Start the Ollama server**:
   - In a separate terminal:
     ```bash
     ollama serve
     ```

9. **Launch the API**:

   ```bash
   uvicorn app.api.main:app --host 0.0.0.0 --port 8000
   ```

## 🚀 Usage

### Via the API

Send a POST request to the `/rag` endpoint:

```bash
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d '{"query": "Quelle est la capitale de la France ?"}'
```

Expected response:

```json
{
  "result": "La capitale de la France est Paris.",
  "sources": [],
  "status": "success"
}
```

Access the Swagger UI at `http://localhost:8000/docs` for interactive testing.

### Via Browser

Visit `http://localhost:8000/docs` to test the API using the Swagger interface.

## 🧪 Tests

Run unit tests:

```bash
pytest tests/test_rag.py
```

## ⚠️ Troubleshooting

- **Memory Error with Mistral**:
  - Error: `model requires more system memory (5.5 GiB) than is available (3.6 GiB)`.
  - Solution: Use a lighter model like `phi3` or `tinyllama`. Update `app/chains/rag_chain.py` and `app/models/llm_factory.py`:
    ```python
    llm = OllamaLLM(model="phi3", ...)
    ```
    Then pull the model:
    ```bash
    ollama pull phi3
    ```
  - Alternatively, increase RAM or swap space:
    - **Windows**: Adjust virtual memory in System Properties → Advanced → Performance → Virtual Memory.
    - **Linux**: Increase swap with `sudo fallocate -l 8G /swapfile` and related commands.

- **Ollama Server Not Running**:
  - Ensure `ollama serve` is running before starting the API.

- **No PDFs Found**:
  - Verify that `app/data/pdfs/` contains valid PDF files. Create the directory if missing:
    ```bash
    mkdir -p app/data/pdfs
    ```

- **GPU Not Detected**:
  - If no NVIDIA GPU is available, the system defaults to CPU. Ensure CUDA drivers are installed if using a compatible GPU.

## 📞 Contact

For questions or issues, contact the maintainers:

- Azzedine ZEMMARI (azzedinezemmari@gmail.com)
- Adama KOMI (adamakomi15@gmail.com)

Or open an issue on [GitHub](https://github.com/adamaKomi/rag).

---

⭐ **Star this project on GitHub if you find it useful!**