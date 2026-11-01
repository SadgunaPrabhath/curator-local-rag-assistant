"""
Offline tests for the ingest -> index -> retrieve path.

These use a real (temporary) ChromaDB but replace the neural embedder with a
tiny deterministic word-hashing embedder, so they run in a second, need no model
downloads and no Ollama, and still prove the plumbing works: chunking, page
tags, caching, document filters, deletion and ranking.
"""

import hashlib
import io
import math

import pytest
from pypdf import PdfWriter

import rag_engine as re_mod
from rag_engine import RagEngine, RagError, count_pdf_pages, extract_pages_from_pdf

DIM = 64


def _fake_embed(texts, progress_callback=None):
    """Bag-of-words hashing embedder: shared words -> similar vectors."""
    vectors = []
    for text in texts:
        v = [0.0] * DIM
        for word in text.lower().split():
            word = word.strip(".,!?;:()\"'")
            if word:
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
                v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vectors.append([x / norm for x in v])
    if progress_callback:
        progress_callback(len(texts), len(texts))
    return vectors


SOLAR = ("Photovoltaic cells convert sunlight into electricity. "
         "Each solar panel is built from silicon wafers.")
FINANCE = ("Annual revenue grew by twelve percent this year. "
           "Operating costs fell during the third quarter.")
BIRDS = ("Migratory birds navigate using the earth magnetic field. "
         "Many species travel thousands of kilometres each autumn.")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "embed_texts", _fake_embed)
    monkeypatch.setattr(re_mod, "get_reranker", lambda: None)
    return RagEngine(persist_dir=str(tmp_path / "db"))


def _fake_pdf(monkeypatch, pages_by_bytes):
    """Make extract_pages_from_pdf / count_pdf_pages return canned pages."""
    calls = {"extract": 0}

    def fake_extract(pdf_source, max_pages=60):
        calls["extract"] += 1
        pages = pages_by_bytes[pdf_source]
        return pages[:max_pages], len(pages)

    monkeypatch.setattr(re_mod, "extract_pages_from_pdf", fake_extract)
    monkeypatch.setattr(re_mod, "count_pdf_pages",
                        lambda src: len(pages_by_bytes[src]))
    return calls


def test_ingest_lists_document_with_chunks(engine, monkeypatch):
    _fake_pdf(monkeypatch, {b"a": [(1, SOLAR), (2, FINANCE)]})
    info = engine.ingest_pdf(b"a", "energy.pdf")
    assert info["cached"] is False
    assert info["chunks"] >= 2 and info["pages_read"] == 2
    docs = engine.list_documents()
    assert len(docs) == 1 and docs[0]["source"] == "energy.pdf"
    assert docs[0]["chunks"] == info["chunks"]


def test_reingesting_same_file_is_cached_and_skips_extraction(engine, monkeypatch):
    calls = _fake_pdf(monkeypatch, {b"a": [(1, SOLAR)]})
    engine.ingest_pdf(b"a", "a.pdf")
    again = engine.ingest_pdf(b"a", "a.pdf")
    assert again["cached"] is True
    assert calls["extract"] == 1          # second call never re-parsed the PDF
    assert len(engine.list_documents()) == 1


def test_retrieve_returns_the_right_page_first(engine, monkeypatch):
    _fake_pdf(monkeypatch, {b"a": [(1, SOLAR), (2, FINANCE), (3, BIRDS)]})
    engine.ingest_pdf(b"a", "mixed.pdf")
    top = engine.retrieve("how do photovoltaic cells convert sunlight", [], "",
                          top_k=3)
    assert top[0]["page"] == 1 and top[0]["source"] == "mixed.pdf"
    top = engine.retrieve("migratory birds magnetic field", [], "", top_k=3)
    assert top[0]["page"] == 3


def test_doc_filter_limits_search_to_selected_documents(engine, monkeypatch):
    _fake_pdf(monkeypatch, {b"a": [(1, SOLAR)], b"b": [(1, BIRDS)]})
    a = engine.ingest_pdf(b"a", "a.pdf")["doc"]
    b = engine.ingest_pdf(b"b", "b.pdf")["doc"]
    only_b = engine.retrieve("photovoltaic sunlight", [], "", doc_filter=[b],
                             top_k=3)
    assert {c["source"] for c in only_b} == {"b.pdf"}
    only_a = engine.retrieve("photovoltaic sunlight", [], "", doc_filter=[a],
                             top_k=3)
    assert {c["source"] for c in only_a} == {"a.pdf"}


def test_delete_document_removes_it_from_search(engine, monkeypatch):
    _fake_pdf(monkeypatch, {b"a": [(1, SOLAR)], b"b": [(1, BIRDS)]})
    a = engine.ingest_pdf(b"a", "a.pdf")["doc"]
    engine.ingest_pdf(b"b", "b.pdf")
    engine.delete_document(a)
    assert [d["source"] for d in engine.list_documents()] == ["b.pdf"]
    hits = engine.retrieve("photovoltaic sunlight", [], "", top_k=3)
    assert all(h["source"] == "b.pdf" for h in hits)


def test_pdf_without_text_raises_clear_error(engine, monkeypatch):
    _fake_pdf(monkeypatch, {b"scan": [(1, "   "), (2, "")]})
    with pytest.raises(RagError, match="scanned image"):
        engine.ingest_pdf(b"scan", "scan.pdf")
    assert engine.list_documents() == []


def test_cached_ingest_reports_truncation_from_page_limit(engine, monkeypatch):
    _fake_pdf(monkeypatch, {b"long": [(i, SOLAR) for i in range(1, 4)]})
    engine.ingest_pdf(b"long", "long.pdf")
    monkeypatch.setattr(re_mod, "MAX_PAGES", 2)
    again = engine.ingest_pdf(b"long", "long.pdf")
    assert again["cached"] is True
    assert again["pages_read"] == 2 and again["total_pages"] == 3
    assert again["truncated"] is True


# --- real PDF parsing (no text, but real files) ----------------------------
def _blank_pdf(n_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_pages_respects_max_pages_and_reports_total():
    pages, total = extract_pages_from_pdf(_blank_pdf(5), max_pages=2)
    assert total == 5 and [p for p, _ in pages] == [1, 2]


def test_count_pdf_pages_matches_real_file():
    assert count_pdf_pages(_blank_pdf(4)) == 4


def test_unreadable_pdf_raises_ragerror():
    with pytest.raises(RagError):
        extract_pages_from_pdf(b"this is not a pdf")
    with pytest.raises(RagError):
        count_pdf_pages(b"this is not a pdf")


# --- configuration helpers --------------------------------------------------
def test_env_int_parses_and_falls_back(monkeypatch):
    monkeypatch.setenv("CURATOR_TEST_INT", "25")
    assert re_mod._env_int("CURATOR_TEST_INT", 60) == 25
    monkeypatch.setenv("CURATOR_TEST_INT", "abc")
    assert re_mod._env_int("CURATOR_TEST_INT", 60) == 60
    monkeypatch.setenv("CURATOR_TEST_INT", "-3")
    assert re_mod._env_int("CURATOR_TEST_INT", 60) == 60
    monkeypatch.delenv("CURATOR_TEST_INT")
    assert re_mod._env_int("CURATOR_TEST_INT", 60) == 60
