"""Tests for the retrieval-evaluation metric helpers in evaluate.py."""

import pytest

from evaluate import first_correct_rank, load_gold, summarize


def test_first_correct_rank_finds_position():
    ranked = [("a.pdf", 3), ("a.pdf", 7), ("b.pdf", 7)]
    assert first_correct_rank(ranked, ("a.pdf", 7)) == 2
    assert first_correct_rank(ranked, ("b.pdf", 7)) == 3


def test_first_correct_rank_requires_source_and_page_to_match():
    ranked = [("a.pdf", 7)]
    assert first_correct_rank(ranked, ("b.pdf", 7)) is None
    assert first_correct_rank(ranked, ("a.pdf", 8)) is None


def test_summarize_hit_rates_and_mrr():
    # ranks: perfect, second, missed, found beyond k
    out = summarize([1, 2, None, 9], k=5)
    assert out["questions"] == 4
    assert out["hit@1"] == pytest.approx(0.25)
    assert out["hit@5"] == pytest.approx(0.5)
    assert out["mrr"] == pytest.approx((1 + 0.5) / 4)


def test_summarize_empty_is_safe():
    assert summarize([], k=5)["questions"] == 0


def test_load_gold_reads_jsonl_and_skips_blank_lines(tmp_path):
    f = tmp_path / "gold.jsonl"
    f.write_text('{"question": "Q1?", "source": "a.pdf", "page": "3"}\n\n'
                 '{"question": "Q2?", "source": "b.pdf", "page": 9}\n',
                 encoding="utf-8")
    rows = load_gold(f)
    assert rows[0] == {"question": "Q1?", "source": "a.pdf", "page": 3}
    assert len(rows) == 2


def test_load_gold_reports_bad_line(tmp_path):
    f = tmp_path / "gold.jsonl"
    f.write_text('{"question": "Q1?"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        load_gold(f)
