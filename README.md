# Curator — Local RAG Document Assistant

Curator is a local-first app for exploring PDF documents. Add your files, ask questions across one or more sources, and review answers alongside the passages and page references they came from.

The app uses retrieval-augmented generation (RAG): it extracts and indexes document text, retrieves relevant passages for each question, and sends that context to a locally running language model. Curator does not require a cloud API key.

## Features

- Upload and search across multiple PDF files.
- Ask questions in a streaming chat and continue with follow-up questions.
- Review source passages with document names and page references.
- Filter which documents are included in a search, or remove a source from the index.
- Create document overviews, summaries, study guides, FAQs, timelines, and briefings.
- Generate text, tables, and charts when the source content supports them.
- Save notes and download generated material as Markdown.
- Keep the vector index on disk between app restarts.

## How it works

```text
PDF files
   │
   ▼
Text extraction and page tagging
   │
   ▼
Sentence-aware chunking
   │
   ▼
Embeddings ──► ChromaDB vector index
                   │
Question ──► query expansion and retrieval
                   │
                   ▼
             Cross-encoder reranking
                   │
                   ▼
           Local Ollama model
                   │
                   ▼
          Answer with citations
```

## Technology

| Area | Technology |
| --- | --- |
| App and interface | Python, Streamlit |
| Document parsing | pypdf |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector search | ChromaDB |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Text generation | Ollama with `llama3.2` |
| Tables and charts | Pandas, Plotly |

## Run locally

You need Python 3.11 or newer and [Ollama](https://ollama.com/download). Curator downloads its embedding and reranking models the first time they are needed, so an internet connection is required for initial setup. After the model files are available, document processing and inference run locally.

### Windows PowerShell

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull llama3.2
python -m streamlit run app.py
```

If PowerShell prevents virtual-environment activation, call the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull llama3.2
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually <http://localhost:8501>. Keep Ollama running while using the app.

## Use Curator

1. Add one or more PDFs in the **Sources** panel. Curator extracts text and indexes it for search.
2. Select the documents you want included in your questions.
3. Ask a question in **Chat**. Review the answer, source passages, and page references; use follow-up suggestions or save useful notes.
4. Use **Studio** to create an overview or a document-based artifact. Generated Markdown can be downloaded from the app.

Each PDF is limited to the first 60 pages during ingestion. Curator extracts embedded PDF text; scanned pages without a text layer need OCR before they can be searched.

## Local data

The ChromaDB index is stored in `chroma_db/` relative to the directory where the app is launched. That folder is ignored by Git because it contains the user's indexed documents and should not be committed. Removing a source in the app deletes its indexed content; deleting `chroma_db/` resets the full local index.

Curator sends prompts to the Ollama service configured on the same machine and runs embeddings and reranking locally. The app has no hosted backend or cloud API integration. The first model download is a separate setup step and requires network access.

## Project files

```text
app.py             Streamlit interface and session workflow
rag_engine.py      PDF ingestion, retrieval, citations, and generation
requirements.txt   Python dependencies
tests/             Offline tests for the RAG engine's pure logic
```

## Run the offline tests

```powershell
python -m pytest -q
```

The tests cover parsing, chunking, numeric grounding, confidence scoring, and mocked answer orchestration. They do not require Ollama or downloading the language models.
