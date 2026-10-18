"""Measure semantic-cache routing against labeled paraphrases and near misses.

Run with the same Python environment as the app:
    python benchmark_cache.py

This measures embedding plus in-memory cache lookup only. It does not call Groq
or measure end-to-end LLM miss latency.
"""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import Main


# Expected answers must match exactly: a hit to the wrong seed is a false hit.
PM_INDIA = "PM of INDIA is Narendra Modi"
PRESIDENT_US = "president of US is Trump"
PRESIDENT_INDIA = "president of INDIA is Murmu"

CASES = [
    (PM_INDIA, "Who is India's prime minister?"),
    (PM_INDIA, "Who currently serves as prime minister of India?"),
    (PM_INDIA, "Tell me the name of the prime minister of India."),
    (PM_INDIA, "Who leads the Indian government as prime minister?"),
    (PM_INDIA, "What is the name of India's PM?"),
    (PM_INDIA, "Who holds the office of prime minister in India?"),
    (PM_INDIA, "Can you tell me who the Indian PM is?"),
    (PRESIDENT_US, "Who is the president of the United States?"),
    (PRESIDENT_US, "Who currently holds the US presidency?"),
    (PRESIDENT_US, "Tell me the name of the American president."),
    (PRESIDENT_US, "Who leads the United States as president?"),
    (PRESIDENT_US, "What is the name of the US president?"),
    (PRESIDENT_US, "Who holds the office of president in America?"),
    (PRESIDENT_INDIA, "Who is India's president?"),
    (PRESIDENT_INDIA, "Who currently serves as president of India?"),
    (PRESIDENT_INDIA, "Tell me the name of the Indian president."),
    (PRESIDENT_INDIA, "Who holds the office of president in India?"),
    (PRESIDENT_INDIA, "What is the name of India's head of state?"),
    (PRESIDENT_INDIA, "Can you tell me who the president of India is?"),
    (PRESIDENT_INDIA, "Who is the Indian head of state?"),
    (None, "Who was India's prime minister before Narendra Modi?"),
    (None, "Who is the prime minister of Pakistan?"),
    (None, "Who is the current president of India?"),
    (None, "When did Narendra Modi become prime minister?"),
    (None, "What are the duties of India's prime minister?"),
    (None, "Who was the US president before Donald Trump?"),
    (None, "Who is the vice president of the United States?"),
    (None, "When is the next US presidential election?"),
    (None, "Who won the last US presidential election?"),
    (None, "What powers does the US president have?"),
    (None, "Who is the prime minister of the United States?"),
    (None, "Who was India's president before Droupadi Murmu?"),
    (None, "Who is the vice president of India?"),
    (None, "When was Droupadi Murmu elected president?"),
    (None, "What is the role of India's president?"),
    (None, "How is the president of India elected?"),
    (None, "Who appoints the prime minister of India?"),
    (None, "Who is the president of Pakistan?"),
    (None, "Who is the president of the Indian National Congress?"),
    (None, "What is the difference between India's president and prime minister?"),
]

REPETITIONS = 5
OUTPUT_PATH = Path(__file__).with_name("cache_benchmark_results.json")


def run_case(expected_answer: str | None, question: str) -> dict:
    query = Main.embed(question)
    # Warm the path so the reported samples are not dominated by one-time setup.
    Main.trusted.lookup(query, time.time())

    samples_ms = []
    actual_answer = None
    score = 0.0
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        query = Main.embed(question)
        actual_answer, score = Main.trusted.lookup(query, time.time())
        samples_ms.append((time.perf_counter() - started) * 1000)
    actual_hit = actual_answer is not None
    if expected_answer is None:
        correct = not actual_hit
        outcome = "false_hit" if actual_hit else "true_miss"
    elif actual_answer == expected_answer:
        correct = True
        outcome = "correct_hit"
    elif actual_hit:
        correct = False
        outcome = "wrong_answer_hit"
    else:
        correct = False
        outcome = "false_negative"

    return {
        "expected": "hit" if expected_answer is not None else "miss",
        "actual": outcome,
        "correct": correct,
        "score": round(score, 4),
        "latency_ms_median": round(statistics.median(samples_ms), 3),
        "question": question,
        "expected_answer": expected_answer,
        "matched_answer": actual_answer,
    }


def main() -> None:
    if len(CASES) < 30 or len(CASES) > 50:
        raise ValueError(f"Expected 30-50 cases; found {len(CASES)}.")

    results = [run_case(expected_answer, question) for expected_answer, question in CASES]
    positives = [result for result in results if result["expected"] == "hit"]
    negatives = [result for result in results if result["expected"] == "miss"]
    correct_hits = sum(result["actual"] == "correct_hit" for result in positives)
    false_negatives = sum(result["actual"] == "false_negative" for result in positives)
    wrong_answer_hits = sum(result["actual"] == "wrong_answer_hit" for result in positives)
    false_hits = sum(result["actual"] == "false_hit" for result in negatives)
    true_misses = sum(result["actual"] == "true_miss" for result in negatives)
    actual_hits = [
        r["latency_ms_median"]
        for r in results
        if r["actual"] in {"correct_hit", "wrong_answer_hit", "false_hit"}
    ]
    actual_misses = [
        r["latency_ms_median"]
        for r in results
        if r["actual"] in {"false_negative", "true_miss"}
    ]

    summary = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": Main.SIMILARITY_THRESHOLD,
        "seed_cache_entries": len(Main.trusted.entries),
        "case_count": len(results),
        "expected_paraphrases": len(positives),
        "expected_near_misses": len(negatives),
        "correct_answer_hits": correct_hits,
        "false_negatives": false_negatives,
        "wrong_answer_hits": wrong_answer_hits,
        "near_miss_false_hits": false_hits,
        "total_false_hits": wrong_answer_hits + false_hits,
        "true_misses": true_misses,
        "paraphrase_correct_hit_rate": round(correct_hits / len(positives), 4),
        "near_miss_false_hit_rate": round(false_hits / len(negatives), 4),
        "cache_answer_precision": round(correct_hits / len(actual_hits), 4)
        if actual_hits
        else 0.0,
        "cache_hit_lookup_latency_ms_median": round(statistics.median(actual_hits), 3)
        if actual_hits
        else None,
        "cache_miss_lookup_latency_ms_median": round(statistics.median(actual_misses), 3)
        if actual_misses
        else None,
        "latency_scope": "embedding plus in-memory cache lookup; excludes Groq/LLM latency",
        "results": results,
    }

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"\nPer-case details written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
