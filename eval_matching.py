"""Measure question-matching quality with the REAL models, then pick thresholds from data.

    python eval_matching.py                 # uses the built-in labelled pairs below
    python eval_matching.py my_pairs.csv    # uses your own pairs (columns: stored,new,label; label 1 = same, 0 = different)

The first run downloads the embedding model and the reranker. No API key is needed.

Two numbers matter, and they are not equally important:
  HIT RATE        of the pairs that SHOULD match, how many did? (a miss only costs a slower answer)
  FALSE-HIT RATE  of the pairs that should NOT match, how many did? (a false hit serves a WRONG answer)
Pick settings with the false-hit rate at (or very near) zero first, then take the best hit rate.
"""
import sys

import numpy as np

import Main  # loads the real models, exactly as the service does

# (stored question, new question). The new question SHOULD be answered from the stored one.
POSITIVES = [
    ("Who wrote Hamlet?", "By whom was Hamlet written?"),
    ("Who discovered penicillin?", "By whom was penicillin discovered?"),
    ("Who painted the Mona Lisa?", "Who was the Mona Lisa painted by?"),
    ("Who invented the telephone?", "By whom was the telephone invented?"),
    ("Who built the Eiffel Tower?", "Who was the Eiffel Tower built by?"),
    ("How do plants carry out photosynthesis?", "How is photosynthesis carried out by plants?"),
    ("What is the capital of France?", "Which city is the capital of France?"),
    ("What is the capital of France?", "Name the capital of France."),
    ("How do I reset my password?", "What are the steps to reset a password?"),
    ("How can I change my password?", "What is the way to change my password?"),
    ("What is your return policy?", "How do returns work?"),
    ("How long does shipping take?", "What is the delivery time?"),
    ("What payment methods do you accept?", "Which payment options are available?"),
    ("Do you ship internationally?", "Is international shipping available?"),
    ("How can I contact customer support?", "What is the way to reach customer support?"),
    ("What are your opening hours?", "When are you open?"),
    ("What is the boiling point of water?", "At what temperature does water boil?"),
    ("How many days are in a leap year?", "How many days does a leap year have?"),
    ("Who wrote Pride and Prejudice?", "Who is the author of Pride and Prejudice?"),
    ("Why does it rain?", "What causes rain?"),
    ("How do I cancel my subscription?", "What is the process for cancelling a subscription?"),
    ("Is the Great Wall of China visible from space?", "Can the Great Wall of China be seen from space?"),
    ("Who was the first person to walk on the moon?", "Who first walked on the moon?"),
    ("When was the Declaration of Independence signed?", "In what year was the Declaration of Independence signed?"),
    ("What does CPU stand for?", "What is the full form of CPU?"),
    ("Why did the Roman Empire fall?", "What caused the fall of the Roman Empire?"),
    ("Can I get a refund if I am not satisfied?", "Do you offer refunds for unsatisfied customers?"),
    ("Where is the Louvre located?", "In which city is the Louvre located?"),
    ("What is the speed of light?", "How fast does light travel?"),
    ("How do I update my billing address?", "Where can I change the billing address on my account?"),
]

# Near-misses: they look alike but have DIFFERENT answers. None of these may match.
NEGATIVES = [
    ("What is the capital of France?", "What is the capital of Spain?"),
    ("Who wrote Hamlet?", "Who wrote Macbeth?"),
    ("Who wrote Hamlet?", "By whom was Macbeth written?"),
    ("Who wrote Hamlet?", "Who translated Hamlet into German?"),
    ("How do I reset my password?", "How do I reset my router?"),
    ("How do I reset my password?", "How do I delete my account?"),
    ("What is your return policy?", "What is your privacy policy?"),
    ("How long does shipping take?", "How much does shipping cost?"),
    ("Do you ship internationally?", "Do you ship to Canada?"),
    ("What is the boiling point of water?", "What is the freezing point of water?"),
    ("What is the boiling point of water at sea level?", "What is the boiling point of water at 3000 meters?"),
    ("How many days are in a leap year?", "How many days are in February in a leap year?"),
    ("Is the Great Wall of China visible from space?", "Is the Great Wall of China the longest wall in the world?"),
    ("Who was the first person to walk on the moon?", "Who was the last person to walk on the moon?"),
    ("When was the Declaration of Independence signed?", "When was the Constitution signed?"),
    ("What does CPU stand for?", "What does GPU stand for?"),
    ("Can I get a refund within 30 days?", "Can I get a refund within 60 days?"),
    ("Is the product safe for children?", "Is the product unsafe for children?"),
    ("Do you accept credit cards?", "Do you accept cash?"),
    ("What are your opening hours on weekdays?", "What are your opening hours on weekends?"),
    ("How do I cancel my subscription?", "How do I upgrade my subscription?"),
    ("Who invented the telephone?", "Who invented the telegraph?"),
    ("What causes rain?", "What causes thunder?"),
    ("What is the speed of light?", "What is the speed of sound?"),
    ("Where is the Louvre located?", "Where is the Prado located?"),
    ("Who painted the Mona Lisa?", "Who stole the Mona Lisa in 1911?"),
    ("How do I update my billing address?", "How do I update my shipping address?"),
    ("Why did the Roman Empire fall?", "Why did the Byzantine Empire fall?"),
    ("What payment methods do you accept?", "What shipping methods do you offer?"),
    ("How do I contact customer support?", "How do I contact sales?"),
    ("Who discovered penicillin?", "Who discovered insulin?"),
]


def load_pairs():
    if len(sys.argv) < 2:
        return POSITIVES, NEGATIVES
    import csv

    pos, neg = [], []
    with open(sys.argv[1], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            same = row["label"].strip().lower() in ("1", "same", "true", "yes")
            (pos if same else neg).append((row["stored"], row["new"]))
    return pos, neg


def score(pairs):
    stored = [a for a, _ in pairs]
    new = [b for _, b in pairs]
    ea = Main.embedder.encode(stored, normalize_embeddings=True)
    eb = Main.embedder.encode(new, normalize_embeddings=True)
    bi = (ea * eb).sum(axis=1)
    cross = None
    if Main.reranker is not None:
        # Same argument order as QuestionMatcher.decide: (new question, stored question).
        cross = np.asarray(Main.reranker.predict([(b, a) for a, b in pairs]), dtype=float)
    nums = np.array([Main.numbers_match(a, b) for a, b in pairs])
    return bi, cross, nums


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    pos, neg = load_pairs()
    if not pos or not neg:
        sys.exit("Need at least one pair with label 1 and one with label 0.")
    pbi, pcross, pnums = score(pos)
    nbi, ncross, nnums = score(neg)
    print(f"{len(pos)} pairs that should match, {len(neg)} near-misses that should not\n")

    print("A) Embedding similarity only (how matching worked before)")
    print("   threshold   hit rate   false-hit rate")
    for t in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95):
        print(f"   {t:8.2f}   {pct((pbi >= t).mean())}    {pct((nbi >= t).mean())}")

    if pcross is None:
        print("\nThe reranker is switched off (RERANKER_MODEL=none), so there is no two-stage table.")
        return

    allc = np.concatenate([pcross, ncross])
    if allc.min() < 0 or allc.max() > 1:
        print(
            "\nWARNING: reranker scores fall outside 0-1, so they look like raw logits. Apply a sigmoid to "
            "them in QuestionMatcher.decide before comparing with MATCH_THRESHOLD, then rerun."
        )

    recalls = (0.35, 0.45, 0.55, 0.65)
    matches = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    grid = []
    for r in recalls:
        for m in matches:
            hits = int(((pbi >= r) & (pcross >= m) & pnums).sum())
            false = int(((nbi >= r) & (ncross >= m) & nnums).sum())
            grid.append((r, m, hits, false))

    print("\nB) Two stages: embedding shortlist (RECALL) then reranker decision (MATCH) then numeric guard")
    print("   hit rate:        columns = MATCH_THRESHOLD    | false-hit rate: same columns")
    print("   RECALL   " + " ".join(f"{m:6.2f}" for m in matches) + "    |  " + " ".join(f"{m:6.2f}" for m in matches))
    for r in recalls:
        row = [g for g in grid if g[0] == r]
        left = " ".join(pct(g[2] / len(pos)) for g in row)
        right = " ".join(pct(g[3] / len(neg)) for g in row)
        print(f"   {r:6.2f}   {left}    |  {right}")

    useful = [g for g in grid if g[2] > 0]
    if not useful:
        print("\nNo setting in this grid produces any hit. Try a different RERANKER_MODEL or EMBED_MODEL.")
        return
    r, m, hits, false = min(useful, key=lambda g: (g[3], -g[2], -g[1], g[0]))
    print(f"\nSuggested starting point: RECALL_THRESHOLD={r}  MATCH_THRESHOLD={m}")
    print(f"   hit rate {pct(hits / len(pos))}, false-hit rate {pct(false / len(neg))}")
    if false == 0:
        print(
            f"   0 false hits in {len(neg)} near-misses only shows the true rate is probably below "
            f"about {100 * 3 / len(neg):.0f}% (rule of three). Add more near-misses from your own traffic."
        )

    print("\nPairs that should match but do not, at this setting (cost: a slower answer):")
    for (a, b), bi, cr, ok in zip(pos, pbi, pcross, pnums):
        if not (bi >= r and cr >= m and ok):
            why = "embedding below recall floor" if bi < r else ("numbers differ" if not ok else "reranker too low")
            print(f"   {bi:.2f}/{cr:.2f}  {a!r} ~ {b!r}   [{why}]")
    print("Pairs that match but should not, at this setting (cost: a WRONG answer):")
    found = False
    for (a, b), bi, cr, ok in zip(neg, nbi, ncross, nnums):
        if bi >= r and cr >= m and ok:
            found = True
            print(f"   {bi:.2f}/{cr:.2f}  {a!r} ~ {b!r}")
    if not found:
        print("   none")
    print("\n(score columns are embedding/reranker)")
    print(f"Use them with:  export RECALL_THRESHOLD={r} MATCH_THRESHOLD={m}")


main()
