"""
evaluate.py - measure how well Curator's retrieval finds the right page
------------------------------------------------------------------------
Retrieval quality is what decides answer quality in a RAG app, so this script
turns it into numbers you can track as you change chunking, models or settings.

You provide a folder of PDFs and a small "gold" file with one JSON object per
line: a question plus the PDF and page that actually contains the answer.

    {"question": "What was the 2023 revenue?", "source": "report.pdf", "page": 12}

Run it (needs the same setup as the app, but NOT Ollama):

    python evaluate.py --pdfs ./my_pdfs --gold gold.jsonl --k 5

It indexes the PDFs into a throwaway database, retrieves the top-k chunks for
each question (vector search + cross-encoder rerank, the same code path the app
uses) and reports:

    hit@1   share of questions whose best-ranked chunk is on the correct page
    hit@k   share of questions with the correct page anywhere in the top k
    MRR     mean reciprocal rank of the first correct page (1.0 is perfect)

A page counts as correct when both its source file name and page number match.
Write 20 to 30 questions about your own documents for numbers worth quoting.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


# ---- pure metric helpers (unit-tested, no models needed) ------------------
def first_correct_rank(ranked: list[tuple[str, int]],
                       expected: tuple[str, int]) -> int | None:
    """1-based rank of the expected (source, page) in `ranked`, or None."""
    for i, item in enumerate(ranked, start=1):
        if item == expected:
            return i
    return None


def summarize(ranks: list[int | None], k: int) -> dict:
    """Aggregate per-question ranks into hit@1, hit@k and MRR."""
    n = len(ranks)
    if n == 0:
        return {"questions": 0, "hit@1": 0.0, f"hit@{k}": 0.0, "mrr": 0.0}
    hit1 = sum(1 for r in ranks if r == 1) / n
    hitk = sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1 / r for r in ranks if r is not None and r <= k) / n
    return {"questions": n, "hit@1": hit1, f"hit@{k}": hitk, "mrr": mrr}


def load_gold(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            rows.append({"question": str(row["question"]),
                         "source": str(row["source"]),
                         "page": int(row["page"])})
        except (ValueError, KeyError) as e:
            raise SystemExit(f"{path}:{lineno}: each line needs question, "
                             f"source and page ({e})")
    return rows


# ---- main -----------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdfs", required=True, type=Path, help="folder of PDFs")
    ap.add_argument("--gold", required=True, type=Path, help="gold .jsonl file")
    ap.add_argument("--k", type=int, default=5, help="chunks to retrieve")
    args = ap.parse_args(argv)

    from rag_engine import RagEngine  # imported late so --help is instant

    pdf_paths = sorted(args.pdfs.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {args.pdfs}")
    gold = load_gold(args.gold)
    if not gold:
        raise SystemExit("The gold file has no questions.")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = RagEngine(persist_dir=tmp)
        for p in pdf_paths:
            info = engine.ingest_pdf(p.read_bytes(), p.name)
            print(f"indexed {p.name}: {info['chunks']} chunks, "
                  f"{info['pages_read']}/{info['total_pages']} pages")

        ranks: list[int | None] = []
        misses = []
        for row in gold:
            chunks = engine.retrieve(row["question"], [], "", top_k=args.k)
            ranked = [(c["source"], c["page"]) for c in chunks]
            rank = first_correct_rank(ranked, (row["source"], row["page"]))
            ranks.append(rank)
            if rank is None:
                misses.append((row, ranked[:3]))

    result = summarize(ranks, args.k)
    print()
    print(f"questions : {result['questions']}")
    print(f"hit@1     : {result['hit@1']:.1%}")
    print(f"hit@{args.k}     : {result[f'hit@{args.k}']:.1%}")
    print(f"MRR       : {result['mrr']:.3f}")
    if misses:
        print(f"\nMissed {len(misses)} question(s) (expected vs. top results):")
        for row, top in misses:
            print(f"  - {row['question']}\n    expected {row['source']} p.{row['page']}"
                  f", got {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
