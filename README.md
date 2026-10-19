# SCPE – Semantic Cache Proxy Engine

Answers repeated questions from a **vetted cache** instead of paying for the LLM again, and protects that cache from wrong or poisoned answers.

Business AIs get asked the same things over and over. Each repeat costs tokens and seconds. This service sits in front of the LLM: if a question with the same *meaning* was already answered and the answer has been vetted, it replies instantly. Otherwise it calls the LLM.

> A wrong cached answer is worse than a slow one, so nothing is cached on first sight.

## What I built

- **Stateless pipeline in LangGraph.** Every question stands alone, with no chat history. The graph decides what gets saved. The LLM never does.
- **Meaning-based matching.** Embeddings shortlist similar stored questions, a cross-encoder checks whether they are duplicates (aimed at paraphrases such as active vs passive voice), and a numeric guard stops "30 days" from matching "60 days".
- **Quarantine → verify → promote.** A new answer must be produced several times and agree with itself. A second LLM call then labels the question and answers (consistent, correct, same for every user, stable, objective, not manipulative), and plain code applies a fixed rule. Anything doubtful is rejected.
- **TTL.** Every cached answer expires, then has to qualify again.
- **Cheap guards.** Answers that hedge, are too long or contain links are never cached, and neither are questions containing obvious attack phrases.
- **Visible behaviour.** Each response reports its source (cache or LLM), cache status, match score and latency, and a small web page shows them.

```
question ─► lookup ──hit──► cached answer
               │ miss
               ▼
          call LLM ─► checks ─► quarantine ─► verify ─► save (with TTL)
```

## Run

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # PowerShell: $env:GROQ_API_KEY="your_key"
python -m uvicorn Main:app --reload
```

Open http://127.0.0.1:8000. The first start downloads the models. Run a single worker.

## Test

```bash
python test_pipeline.py   # 62 logic checks with fake models (no key, no download)
python eval_matching.py   # measures matching quality with the real models and suggests thresholds
```

## Status

Prototype. The cache is in memory, so a restart empties it. Matching thresholds are untuned guesses until you run `eval_matching.py`. There is no authentication, rate limiting or purge endpoint yet. Settings are documented at the top of `Main.py`.

**Stack:** FastAPI, LangGraph, LangChain (Groq), sentence-transformers (MiniLM embeddings and a cross-encoder reranker), NumPy.
