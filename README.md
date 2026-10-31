# 🏛️ Curator — AI-Powered RAG Document Assistant

> Your sources, curated and explained.

Curator is a local, AI-powered document assistant that allows users to upload PDFs, ask questions, and generate document-grounded answers with source citations.

Built using Retrieval-Augmented Generation (RAG), Curator combines semantic search, reranking, and a locally running LLM to provide reliable answers without requiring cloud APIs.

## ✨ Features

- 📚 Upload and manage multiple PDF documents
- 💬 Streaming AI chat with follow-up questions
- 🔎 Source citations with page references
- 🧠 Semantic search using embeddings
- 🎯 Cross-encoder reranking
- 🗃️ Persistent ChromaDB vector storage
- 📊 Text, table, and chart-based responses
- 🎛️ Studio for generating document-based artifacts
- 📝 Summaries and study guides
- 📥 Download generated Markdown files
- 🛡️ Document-grounded responses to reduce hallucinations
- 🔒 Local processing using Ollama

## ⚙️ How It Works

```text
PDF Documents
     │
     ▼
Text Extraction
     │
     ▼
Sentence-Aware Chunking
     │
     ▼
Embedding Generation
     │
     ▼
ChromaDB Vector Storage
     │
     ▼
Query Expansion and Retrieval
     │
     ▼
Cross-Encoder Reranking
     │
     ▼
Local LLM Response
     │
     ▼
Answer + Citations + Visualization
```
