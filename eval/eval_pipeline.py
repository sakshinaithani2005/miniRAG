"""
RAG Evaluation Pipeline using RAGAS.

Measures four core retrieval-augmented generation quality metrics:

  - Faithfulness        : Is the answer grounded in the retrieved context?
  - Answer Relevancy    : Does the answer address the question?
  - Context Precision   : Are the retrieved chunks relevant to the question?
  - Context Recall      : Do the retrieved chunks cover the ground-truth answer?

Usage::

    # Run against the bundled sample dataset (no PDF needed)
    python eval/eval_pipeline.py

    # Index a PDF first, then evaluate
    python eval/eval_pipeline.py --pdf path/to/attention.pdf

    # Save results to a JSON file
    python eval/eval_pipeline.py --pdf paper.pdf --output eval/results/latest_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src is on the path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_env() -> None:
    from dotenv import load_dotenv

    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        load_dotenv(env, override=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="miniRAG evaluation pipeline (RAGAS metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to a PDF to index before evaluating. "
             "If omitted, uses built-in sample Q/A pairs.",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset" / "sample_qa.json"),
        help="Path to Q/A JSON dataset (default: eval/dataset/sample_qa.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results (e.g. eval/results/latest_eval.json)",
    )
    parser.add_argument(
        "--strategy",
        choices=["dense", "hybrid", "mmr"],
        default="hybrid",
        help="Retrieval strategy to evaluate (default: hybrid)",
    )
    return parser.parse_args()


def _index_pdf(pdf_path: Path) -> list[Any]:
    """Index a PDF and return the chunks."""
    print(f"Indexing {pdf_path.name} …")
    from minirag import add_documents_to_vectorstore, get_vectorstore, process_documents

    class _FileWrapper:
        def __init__(self, p: Path):
            self._bytes = p.read_bytes()
            self.name = p.name
            self.type = "application/pdf"

        def getvalue(self) -> bytes:
            return self._bytes

    chunks = process_documents(uploaded_file=_FileWrapper(pdf_path))
    vs = get_vectorstore()
    add_documents_to_vectorstore(chunks, vs)
    print(f"Indexed {len(chunks)} chunks.")
    return chunks


def run_evaluation(
    qa_pairs: list[dict],
    strategy: str = "hybrid",
    corpus: list | None = None,
) -> dict[str, Any]:
    """
    Run RAGAS evaluation over a list of Q/A pairs.

    Args:
        qa_pairs:  List of dicts with keys: question, ground_truth, contexts.
        strategy:  Retrieval strategy name.
        corpus:    Indexed chunks (needed for HYBRID strategy).

    Returns:
        Dict with per-metric scores and aggregate stats.
    """
    from minirag import (
        RetrievalStrategy,
        create_rag_chain,
        create_retriever,
        get_llm,
        get_vectorstore,
        query_rag,
    )

    vs = get_vectorstore()
    llm = get_llm()
    retrieval_strategy = RetrievalStrategy(strategy)
    retriever = create_retriever(vs, corpus=corpus, strategy=retrieval_strategy)
    chain = create_rag_chain(retriever, llm)

    # ── Collect responses ────────────────────────────────────────────────
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    print(f"\nRunning {len(qa_pairs)} queries with '{strategy}' retrieval …\n")

    for i, pair in enumerate(qa_pairs, 1):
        q = pair["question"]
        print(f"  [{i}/{len(qa_pairs)}] {q[:80]}")

        answer, docs, _ = query_rag(
            chain, retriever, q,
            enable_rewrite=True, llm=llm,
        )

        retrieved_contexts = [d.page_content for d in docs]

        questions.append(q)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(pair.get("ground_truth", ""))

    # ── RAGAS evaluation ─────────────────────────────────────────────────
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        print("\nComputing RAGAS metrics …")
        t0 = time.perf_counter()
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        elapsed = time.perf_counter() - t0

        scores = {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall": round(float(result["context_recall"]), 4),
            "evaluation_time_s": round(elapsed, 2),
            "num_questions": len(questions),
            "retrieval_strategy": strategy,
        }

    except ImportError:
        print("WARNING: RAGAS not installed. Computing basic hit-rate metrics instead.")
        print("   Install with: pip install ragas datasets")

        # Fallback: simple grounding check
        from rag_chain import verify_citations

        total_warnings = sum(len(verify_citations(a, len(c))) for a, c in zip(answers, contexts, strict=False))
        scores = {
            "citation_warning_rate": round(total_warnings / len(answers), 4),
            "avg_contexts_retrieved": round(
                sum(len(c) for c in contexts) / len(contexts), 2
            ),
            "num_questions": len(questions),
            "retrieval_strategy": strategy,
            "note": "Install ragas + datasets for full RAGAS metrics",
        }

    return scores


def _print_table(scores: dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"{'Metric':<28} {'Score':>10}")
    print("-" * 55)
    skip = {"num_questions", "retrieval_strategy", "evaluation_time_s", "note"}
    for k, v in scores.items():
        if k not in skip:
            bar = "█" * int(float(v) * 20) if isinstance(v, float) else ""
            print(f"{k:<28} {v:>10.4f}  {bar}")
    print("-" * 55)
    print(f"{'Strategy':<28} {scores.get('retrieval_strategy', '-'):>10}")
    print(f"{'Questions evaluated':<28} {scores.get('num_questions', '-'):>10}")
    if "evaluation_time_s" in scores:
        print(f"{'Eval time':<28} {scores['evaluation_time_s']:>9.1f}s")
    print("=" * 55)


def main() -> int:
    _load_env()
    args = _parse_args()

    from minirag import Config
    if not Config.validate():
        print("Error: Missing API keys.", file=sys.stderr)
        return 1

    # Load Q/A dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    with open(dataset_path) as f:
        qa_pairs = json.load(f)

    # Optionally index a PDF
    corpus = None
    if args.pdf:
        corpus = _index_pdf(Path(args.pdf))

    # Run evaluation
    scores = run_evaluation(qa_pairs, strategy=args.strategy, corpus=corpus)
    _print_table(scores)

    # Optionally save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
