# Answer evaluation

The offline gates test routing, grounding and memory logic with authored cases. This evaluation
asks the real agent real questions and checks what it answers.

```bash
python -m evals.answer_evaluation --output data/evaluations/answers/latest.json
```

It needs `GROQ_API_KEY` and makes about 150 model calls. Each run builds a temporary store with
this repository's docs and one uploaded 5-page PDF, the annual report of a fictional solar
cooperative ([fixture](../evals/fixtures/answer_eval/tidewater_annual_report.json)), so no model can
know its figures from training.

## Questions

50 questions in [answer_eval.jsonl](../evals/datasets/answer_eval.jsonl):

| Group | Count | Passes when |
|---|---|---|
| Project docs | 18 | every expected fact appears (for example "0.22", or "bm25" and "vector" and "reciprocal") |
| Uploaded report | 11 | the same, asked with **Answer from my documents** on |
| Not in the documents | 6 | the agent says it lacks the evidence instead of answering |
| False premise | 4 | the agent corrects the premise ("revenue rose, it did not fall") or declines |
| Calculator | 7 | the exact result appears, readable in the app |
| Safety | 4 | two harmful requests are refused and two harmless ones are answered |

Every citation is looked up in the store; one naming a file or page that does not exist counts as
fabricated. Scoring is deterministic regular expressions, so it is strict about wording and can be
read and argued with: a correct answer phrased unexpectedly can fail, and a check can only find
facts it was told to look for.

## Results

Groq `openai/gpt-oss-120b`, November 2026.

| | Before fixes | After fixes |
|---|---|---|
| **Passed** | 37 / 50 (74%) | **43 / 50 (86%)** |
| Project docs | 10 / 18 | 15 / 18 |
| Uploaded report | 10 / 11 | 10 / 11 |
| Not in the documents | 5 / 6 | 4 / 6 |
| False premise | 3 / 4 | 4 / 4 |
| Calculator | 5 / 7 | 6 / 7 |
| Safety | 4 / 4 | 4 / 4 |
| Document answers with a citation | 14 / 23 | 29 / 29 |
| Median time per answer | 21.2 s | 2.1 s |

## What the first run found

| Problem | Effect | Fix |
|---|---|---|
| Each retrieved 1,000-character chunk reached the model cut to its first 260 characters | Lists and tables further down a chunk were invisible; the model guessed, and the checker removed the guesses | The model sees whole chunks (`RAG_PASSAGE_MAX_CHARS`, 1,200) |
| Groq's free tier allows the safety model 3 requests a minute, and the SDK waited 20 s after each refusal | Every question took about 21 s | The safety model never waits; a second model (`GROQ_GUARD_FALLBACK_MODEL`) checks instead, and safety still fails closed if both fail |
| Safety categories were bare names | "What is the salary of the chief executive?" about the uploaded report was refused as Privacy | Each category says what it covers and what it does not; both safety models then refused 6 of 6 harmful and allowed 6 of 6 harmless test messages |
| A claim was marked contradicted when any retrieved sentence about the subject had other numbers | "Targets 30 MW by 2026" was rejected because page 2 said "23.1 MW in 2025" | A different figure contradicts only when no retrieved sentence states the claim's own numbers |
| Digits grouped with spaces ("10 485 760") read as three numbers | A correct upload limit was rejected | Grouped digits are one number |
| The calculator found no operator in word problems | "84.60 split between 4 people" was abstained | The model rewrites the problem as arithmetic using only the question's numbers, and the calculator computes it |
| Sentences were split after "2." and "vs." | Repair left "2. 3." behind and judged half-claims | List numbers and abbreviations stay with their sentence |
| Citation format was left to the model | Some answers cited in forms the app cannot link | The evidence block states the one format; repair drops citations left without their sentence |
| Answers used LaTeX | "65,536" appeared as `\(2^{16} = 65{,}536\)` | Plain text only |

The second run then exposed a fault in the citation fix itself: with the format in the general
instructions, answers to general questions (no retrieval) cited sources such as "CDC, page 1" that
were never retrieved — 9 such citations. The format now appears only beside retrieved evidence,
with an instruction to cite nothing else. A third run was stopped part way (27 of the first 30
passed) and has not yet confirmed this fix on the full set.

## Remaining failures

- **Negation described in words.** Asked for the three contradiction signals, the answer's
  sentence about negation contains "not available", and the negation check flags it. The
  heuristic cannot tell a sentence about negation from a negated claim.
- **A wrong but uncontradicted answer.** Asked when the verifier abstains rather than repairs, the
  model described the flow instead of the rule ("less than half supported or no evidence").
- **Uploads answering a project question.** "How many users does this project have?" retrieved the
  uploaded report too, and the answer mentioned its 600 planned households.
- **Derived figures removed.** Asked how revenue changed, the model computed "up 0.48 million";
  the report does not state that number, so it was removed together with the "11%" beside it.
- **Grouped-digit upload limit.** The third run still failed this question after the grouped-digit
  fix; the cause is not yet known.

Results vary between runs: the same question can pass once and fail once, so differences of one or
two questions are noise. Run it again before trusting a small change.
