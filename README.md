# Curator — Local RAG Document Assistant

[![tests](https://github.com/SadgunaPrabhath/curator-local-rag-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/SadgunaPrabhath/curator-local-rag-assistant/actions/workflows/tests.yml)

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

## Configuration

The defaults work out of the box. To change them, set environment variables before starting the app (the models can be swapped without editing code):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CURATOR_MODEL` | `llama3.2` | Ollama model used to write answers (run `ollama pull <name>` first) |
| `CURATOR_EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers embedding model |
| `CURATOR_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder used to rerank retrieved passages |
| `CURATOR_MAX_PAGES` | `60` | Pages read from each PDF during ingestion |
| `CURATOR_DB_DIR` | `./chroma_db` | Where the vector index is stored |

```bash
CURATOR_MODEL=llama3.1:8b python -m streamlit run app.py
```

On Windows PowerShell, set the variable first with `$env:CURATOR_MODEL = "llama3.1:8b"`. If you change the embedding model, delete the old index folder, because vectors from different models cannot be mixed.

## Evaluate retrieval quality

Answer quality depends on retrieving the right passage, so Curator includes a small script that measures it. Put some PDFs in a folder and write a `gold.jsonl` file with one question per line, naming the PDF and page that contains the answer:

```json
{"question": "What was the 2023 revenue?", "source": "report.pdf", "page": 12}
```

```bash
python evaluate.py --pdfs ./my_pdfs --gold gold.jsonl --k 5
```

It indexes the PDFs into a temporary database, runs the same retrieval and reranking code the app uses (no Ollama needed), and prints **hit@1**, **hit@k** and **MRR**, plus the questions it missed. Use it to compare chunk sizes, embedding models or rerankers with numbers instead of impressions. Twenty to thirty questions about your own documents give figures worth reporting.

## Limitations

- PDF only, and only the text layer: scanned pages need OCR first.
- Only the first 60 pages of each PDF are read by default (see `CURATOR_MAX_PAGES`).
- Answer quality is bounded by the local model. A small model such as `llama3.2` can still miss details or phrase things loosely, so check the cited pages for anything important.
- The default context window is deliberately small (2048 tokens) to keep generation fast on CPU, so only a few passages are used per answer.
- Retrieval quality has not been published for a benchmark dataset. Run `evaluate.py` on your own documents to measure it.

## Local data

The ChromaDB index is stored in `chroma_db/` relative to the directory where the app is launched. That folder is ignored by Git because it contains the user's indexed documents and should not be committed. Removing a source in the app deletes its indexed content; deleting `chroma_db/` resets the full local index.

Curator sends prompts to the Ollama service configured on the same machine and runs embeddings and reranking locally. The app has no hosted backend or cloud API integration. The first model download is a separate setup step and requires network access.

## Project files

```text
app.py                  Streamlit interface and session workflow
rag_engine.py           PDF ingestion, retrieval, citations, and generation
evaluate.py             Retrieval evaluation (hit@k, MRR) on your own PDFs
requirements.txt        Python dependencies
tests/                  Offline tests for parsing, chunking, indexing and retrieval
examples/hello_ollama.py  Minimal script to check that Ollama works
.github/workflows/      Continuous integration that runs the tests
```

## Run the offline tests

```bash
python -m pytest -q
```

The tests cover parsing, chunking, numeric grounding, confidence scoring, mocked answer orchestration, and the full ingest, index, filter, delete and retrieve path against a temporary ChromaDB. They use a small deterministic stand-in for the embedding model, so they do not require Ollama or downloading any models, and they run on every push through GitHub Actions.
