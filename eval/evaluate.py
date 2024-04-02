"""RAG evaluation script.

Loads an evaluation dataset, runs each question through the RAG pipeline,
and computes faithfulness, answer relevance, and context precision metrics.

Usage:
    python -m eval.evaluate --config config/config.yaml --dataset eval/sample_dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0


@dataclass
class EvalRecord:
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: list[str] = field(default_factory=list)
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    latency_seconds: float = 0.0


@dataclass
class EvalSummary:
    total_questions: int = 0
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_context_precision: float = 0.0
    avg_latency_seconds: float = 0.0
    records: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _sentence_split(text: str) -> list[str]:
    """Naive sentence splitting on period/question/exclamation boundaries."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _word_set(text: str) -> set[str]:
    """Lowercase word set from text."""
    return set(text.lower().split())


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """Heuristic faithfulness: fraction of answer sentences whose words overlap
    substantially with the retrieved context.

    A sentence is considered 'grounded' if at least 40% of its non-stopword tokens
    appear in the combined context text.
    """
    if not answer.strip() or not contexts:
        return 0.0

    context_words = set()
    for ctx in contexts:
        context_words.update(_word_set(ctx))

    # Minimal stopword set to avoid inflating overlap scores
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "if",
        "when",
        "while",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
    }

    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0

    grounded = 0
    for sent in sentences:
        words = _word_set(sent) - stopwords
        if not words:
            grounded += 1  # trivial sentence, count as grounded
            continue
        overlap = words & context_words
        if len(overlap) / len(words) >= 0.4:
            grounded += 1

    return grounded / len(sentences)


def compute_embedding_similarity(text_a: str, text_b: str, embeddings) -> float:
    """Cosine similarity between two texts using a LangChain embeddings model."""
    try:
        vecs = embeddings.embed_documents([text_a, text_b])
        a = np.array(vecs[0])
        b = np.array(vecs[1])
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception:
        logger.warning("embedding_similarity.failed", text_a=text_a[:50], text_b=text_b[:50])
        return 0.0


def compute_answer_relevance(question: str, answer: str, embeddings) -> float:
    """Answer relevance via cosine similarity between question and answer embeddings."""
    if not question.strip() or not answer.strip():
        return 0.0
    return compute_embedding_similarity(question, answer, embeddings)


def compute_context_precision(question: str, contexts: list[str], embeddings) -> float:
    """Context precision: average cosine similarity between question and each context chunk."""
    if not question.strip() or not contexts:
        return 0.0

    scores = []
    for ctx in contexts:
        if ctx.strip():
            scores.append(compute_embedding_similarity(question, ctx, embeddings))

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Pipeline interaction
# ---------------------------------------------------------------------------


def _build_rag_chain(config_path: str):
    """Build the RAG chain from config. Returns (chain, embeddings)."""
    from src.config import load_config
    from src.generation.chain import build_rag_chain
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import get_reranker
    from src.retrieval.vectorstore import get_embeddings, get_vectorstore

    cfg = load_config(config_path)
    embeddings = get_embeddings(cfg.embedding)
    vectorstore = get_vectorstore(cfg.vectorstore, embeddings)
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        config=cfg.retrieval,
    )
    reranker = get_reranker(cfg.retrieval)
    chain = build_rag_chain(
        llm_config=cfg.llm,
        retriever=retriever,
        reranker=reranker,
    )
    return chain, embeddings, retriever


def _run_query(chain, retriever, question: str) -> tuple[str, list[str]]:
    """Run a single question through the RAG chain, returning (answer, contexts)."""
    # Retrieve contexts
    results = retriever.search(question)
    contexts = [r.content for r in results]

    # Generate answer
    response = chain.invoke({"question": question})
    answer = response if isinstance(response, str) else str(response.get("answer", response))

    return answer, contexts


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def load_dataset(dataset_path: str | Path) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset must be a non-empty JSON array: {path}")

    required_keys = {"question", "ground_truth"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Dataset item {i} missing keys: {missing}")

    return data


def evaluate(
    config_path: str,
    dataset_path: str,
    output_path: str | None = None,
) -> EvalSummary:
    """Run full evaluation pipeline.

    Args:
        config_path: Path to config YAML.
        dataset_path: Path to eval dataset JSON.
        output_path: Optional path to write results JSON.

    Returns:
        EvalSummary with per-question records and aggregate metrics.
    """
    dataset = load_dataset(dataset_path)
    chain, embeddings, retriever = _build_rag_chain(config_path)

    records: list[EvalRecord] = []

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("eval.running", question_idx=i + 1, total=len(dataset))
        start = time.time()

        try:
            answer, contexts = _run_query(chain, retriever, question)
        except Exception:
            logger.exception("eval.query_failed", question=question)
            answer = ""
            contexts = []

        latency = time.time() - start

        metrics = EvalMetrics(
            faithfulness=compute_faithfulness(answer, contexts),
            answer_relevance=compute_answer_relevance(question, answer, embeddings),
            context_precision=compute_context_precision(question, contexts, embeddings),
        )

        record = EvalRecord(
            question=question,
            ground_truth=ground_truth,
            generated_answer=answer,
            retrieved_contexts=contexts,
            metrics=metrics,
            latency_seconds=round(latency, 3),
        )
        records.append(record)

    # Aggregate
    summary = EvalSummary(
        total_questions=len(records),
        avg_faithfulness=float(np.mean([r.metrics.faithfulness for r in records])),
        avg_answer_relevance=float(np.mean([r.metrics.answer_relevance for r in records])),
        avg_context_precision=float(np.mean([r.metrics.context_precision for r in records])),
        avg_latency_seconds=float(np.mean([r.latency_seconds for r in records])),
        records=[asdict(r) for r in records],
    )

    # Print formatted table
    _print_results_table(records, summary)

    # Save to file
    out = Path(output_path) if output_path else Path(__file__).parent / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(summary), f, indent=2, default=str)
    logger.info("eval.results_saved", path=str(out))

    return summary


def _print_results_table(records: list[EvalRecord], summary: EvalSummary) -> None:
    """Print a formatted results table to stdout."""
    header = (
        f"{'#':<4} {'Faithfulness':>13} {'Relevance':>10} {'Ctx Prec':>9} {'Latency':>8}  Question"
    )
    sep = "-" * 100

    print("\n" + sep)  # noqa: T201
    print("RAG EVALUATION RESULTS")  # noqa: T201
    print(sep)  # noqa: T201
    print(header)  # noqa: T201
    print(sep)  # noqa: T201

    for i, rec in enumerate(records, 1):
        m = rec.metrics
        q = rec.question[:55] + "..." if len(rec.question) > 58 else rec.question
        print(  # noqa: T201
            f"{i:<4} {m.faithfulness:>13.3f} {m.answer_relevance:>10.3f} "
            f"{m.context_precision:>9.3f} {rec.latency_seconds:>7.2f}s  {q}"
        )

    print(sep)  # noqa: T201
    print(  # noqa: T201
        f"{'AVG':<4} {summary.avg_faithfulness:>13.3f} {summary.avg_answer_relevance:>10.3f} "
        f"{summary.avg_context_precision:>9.3f} {summary.avg_latency_seconds:>7.2f}s"
    )
    print(sep + "\n")  # noqa: T201


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        default="eval/sample_dataset.json",
        help="Path to evaluation dataset JSON (default: eval/sample_dataset.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (default: eval/results.json)",
    )
    args = parser.parse_args()

    try:
        evaluate(
            config_path=args.config,
            dataset_path=args.dataset,
            output_path=args.output,
        )
    except Exception:
        logger.exception("eval.failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
