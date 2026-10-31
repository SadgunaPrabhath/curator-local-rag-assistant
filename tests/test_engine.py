"""
Offline unit tests for the pure logic in rag_engine.
Run:  pytest -q
These never touch Ollama, download the reranker, or hit disk — the heavy models
are avoided by testing the pure functions and monkeypatching where needed.
"""

import pytest

import rag_engine as re_mod
from rag_engine import (
    NOT_FOUND,
    _confidence,
    _numbers_grounded,
    _pack_sentences,
    _safe_parse,
    _split_sentences,
    chunk_pages,
    chunk_text,
)


# --- chunk_text (word-based, backwards-compatible) ------------------------
def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_chunk_text_short_single_chunk():
    assert chunk_text("a b c") == ["a b c"]


def test_chunk_text_no_redundant_tail():
    # 210 words, size 200 / overlap 40 -> exactly 2 chunks, no overlap-only tail
    assert len(chunk_text(" ".join(str(i) for i in range(210)))) == 2


def test_chunk_text_overlap_guard():
    with pytest.raises(ValueError):
        chunk_text("a b c d", chunk_size=3, overlap=3)


def test_chunk_text_overlap_stride():
    ch = chunk_text(" ".join(str(i) for i in range(12)), chunk_size=5, overlap=2)
    # consecutive chunks share exactly `overlap` (=2) words
    a = ch[0].split()
    b = ch[1].split()
    assert a[-2:] == b[:2]


def test_chunk_text_unicode_roundtrip():
    text = "café résumé 日本語 😀 telugu"
    assert chunk_text(text) == [text]


# --- sentence-aware chunking ----------------------------------------------
def test_split_sentences():
    assert _split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_pack_sentences_respects_size_and_overlap():
    sents = [f"word{i} more text here." for i in range(10)]  # 4 words each
    chunks = _pack_sentences(sents, chunk_words=8, overlap_sentences=1)
    assert len(chunks) > 1
    # overlap: last sentence of a chunk starts the next one
    first_last = chunks[0].split(".")[-2].strip() + "."
    assert first_last in chunks[1]


def test_chunk_pages_tags_pages():
    pages = [(1, "Alpha sentence. Beta sentence."), (2, "Gamma sentence.")]
    out = chunk_pages(pages, chunk_words=50, overlap_sentences=0)
    assert all("text" in c and "page" in c for c in out)
    assert {c["page"] for c in out} == {1, 2}


def test_chunk_pages_empty():
    assert chunk_pages([(1, "   "), (2, "")]) == []


# --- structured output parsing --------------------------------------------
def test_safe_parse_valid_chart():
    raw = '{"answer":"x","format":"bar","data":[{"label":"A","value":1}]}'
    p = _safe_parse(raw)
    assert p["format"] == "bar" and p["answer"] == "x"


def test_safe_parse_bad_json_falls_back_to_text():
    p = _safe_parse("not json at all")
    assert p["format"] == "text" and p["answer"] == "not json at all"


def test_safe_parse_invalid_format_downgrades():
    p = _safe_parse('{"answer":"y","format":"scatter3d"}')
    assert p["format"] == "text"


def test_safe_parse_missing_answer_falls_back():
    p = _safe_parse('{"format":"pie"}')
    assert p["format"] == "text"


# --- numeric grounding guard ----------------------------------------------
def test_numbers_grounded_true():
    ctx = "Revenue was 1,200 in 2021 and 1500 in 2022."
    data = [{"label": "2021", "value": 1200}, {"label": "2022", "value": 1500}]
    assert _numbers_grounded(data, ctx) is True


def test_numbers_grounded_false_on_invented_number():
    ctx = "Revenue was 1200 in 2021."
    data = [{"label": "2021", "value": 9999}]
    assert _numbers_grounded(data, ctx) is False


def test_numbers_grounded_false_on_non_list():
    assert _numbers_grounded(None, "anything") is False


def test_numbers_grounded_handles_trailing_period_and_int():
    # "1200." (end of sentence) and integer values must still match
    ctx = "Revenue in 2021 was 1200. It rose to 1500 in 2022."
    data = [{"label": "2021", "value": 1200}, {"label": "2022", "value": 1500}]
    assert _numbers_grounded(data, ctx) is True


def test_numbers_grounded_float_vs_comma():
    ctx = "Total was 1,200,000 last year."
    assert _numbers_grounded([{"label": "x", "value": 1200000}], ctx) is True


# --- confidence mapping ---------------------------------------------------
def test_confidence_high_medium_low():
    assert _confidence([{"score": 5.0}]) == "High"
    assert _confidence([{"score": 0.0}]) == "Medium"
    assert _confidence([{"score": -5.0}]) == "Low"
    assert _confidence([{}]) == "Medium"   # no scores -> neutral


# --- query understanding + generation, with a fake Ollama -----------------
class _FakeResp(dict):
    pass


def test_answer_orchestrates_with_mocked_ollama(monkeypatch):
    """Drive answer() end-to-end without a real model, embedder, or Chroma."""
    import json

    def fake_chat(model, messages, **kw):
        content = messages[-1]["content"]
        if kw.get("format") == "json" and "search engine" in content:
            # _expand_query
            return {"message": {"content": json.dumps(
                {"queries": ["q1", "q2"], "hyde": "h",
                 "needs_clarification": False, "clarifying_question": ""})}}
        if kw.get("format") == "json":
            # _VIZ generation
            return {"message": {"content": json.dumps(
                {"answer": "Grounded answer.", "format": "text",
                 "data": None, "followups": ["next?"]})}}
        return {"message": {"content": "rewritten"}}

    monkeypatch.setattr(re_mod.ollama, "chat", fake_chat)

    # Fake retrieval so we don't touch the embedder/Chroma.
    def fake_retrieve(self, q, expansions, hyde, doc_filter=None, top_k=4):
        return [{"text": "ctx chunk", "source": "doc.pdf", "page": 3, "score": 4.0}]

    monkeypatch.setattr(re_mod.RagEngine, "retrieve", fake_retrieve)
    monkeypatch.setattr(re_mod.RagEngine, "__init__", lambda self: None)

    eng = re_mod.RagEngine()
    out = eng.answer("revenue?", history=[])
    assert out["answer"] == "Grounded answer."
    assert out["format"] == "text"
    assert out["confidence"] == "High"
    assert out["sources"][0]["page"] == 3
    assert out["followups"] == ["next?"]


def test_not_found_constant_stable():
    assert NOT_FOUND == "I couldn't find that in the document."


# --- whole-document query router ------------------------------------------
def test_is_global_query():
    from rag_engine import _is_global_query
    assert _is_global_query("summarize this document")
    assert _is_global_query("What are the main themes?")
    assert _is_global_query("give me a tldr")
    assert not _is_global_query("what was the revenue in 2021?")
    assert not _is_global_query("who is the CEO")
    assert not _is_global_query("")


# --- empty doc_filter means 'no sources', not 'all sources' ---------------
def test_retrieve_empty_filter_returns_empty(monkeypatch):
    monkeypatch.setattr(re_mod.RagEngine, "__init__", lambda self: None)
    eng = re_mod.RagEngine()
    assert eng.retrieve("q", [], "", doc_filter=[]) == []


def test_get_doc_chunks_empty_filter_returns_empty(monkeypatch):
    monkeypatch.setattr(re_mod.RagEngine, "__init__", lambda self: None)
    eng = re_mod.RagEngine()
    assert eng.get_doc_chunks(doc_filter=[]) == []
