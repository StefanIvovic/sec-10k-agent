# Eval questions

| File | Use |
|---|---|
| `dev.jsonl` | Tune against these freely. Scores here are optimistic. |
| `test.jsonl` | Held out. Run it only to report a score, never to decide what to fix. If a test failure leads to a code change, move that question to `dev.jsonl` and write a new one. |

One line per question. A question may appear only once per file, and never in both files (the runner checks).

```
python -m sec_agent.eval.run --split dev
python -m sec_agent.eval.run --split test --run-name test-v1
```

## Line format

```json
{"question": "What was Apple's revenue in fiscal 2026?", "ticker": "AAPL", "expect": "answer", "number": 478834000000, "source": "Total net sales"}
{"question": "What are Apple's reportable segments?", "ticker": "AAPL", "expect": "answer", "contains": ["Americas", "Europe", "Greater China"]}
{"question": "What is Apple's revenue in Brazil?", "ticker": "AAPL", "expect": "not_found", "why": "Brazil is not in the 10-K"}
{"question": "Why did Apple's revenue decline in fiscal 2026?", "ticker": "AAPL", "expect": "correction", "contains": ["increase", "grew"]}
```

| Field | Required | Meaning |
|---|---|---|
| `question` | yes | What the agent is asked. |
| `expect` | yes | `answer`: the agent should answer; saying "not found" fails.<br>`not_found`: the filings don't have it; the agent should say so, and inventing an answer fails.<br>`correction`: the question has a false premise; the agent should say not found or correct it. |
| `ticker` | for `source` | The company the question is about. |
| `number` | no | (`answer`) This value must appear in the answer, within 1%. Base units: dollars, not millions. |
| `contains` | no | (`answer`) Every phrase must appear in the answer, case-insensitive.<br>(`correction`, required) At least one must appear, e.g. "increased". |
| `source` | no | A phrase, or a list of phrases, copied exactly from the filing. Also checks that the search found a paragraph of that company containing one of them. Keep it specific: "Total revenue" appears in dozens of paragraphs. |
| `id`, `why`, `source_item` | no | Notes for you; not scored. |

An `answer` line with no `number` or `contains` passes as long as the agent doesn't say "not found".

## What a run does

- Every line is asked once. `not_found` and `correction` lines are asked 3 times (`--repeats`), because they must hold up every time.
- MLflow gets one table (each run: pass/fail and the reason) and these totals:

| Metric | Meaning |
|---|---|
| `answer_pass_rate` | `answer` lines answered correctly |
| `false_refusal_rate` | `answer` lines where the agent said "not found" |
| `made_up_rate` | `not_found` lines where the agent invented an answer |
| `correction_pass_rate` | false premises refused or corrected |
| `*_consistency` | same result on every repeat |
| `source_found_rate` | lines with `source` where the search found the paragraph |

At the end it prints the failed questions with their line numbers.

## Tips

- Check `number` against the filing or XBRL, not against the agent. `python -m sec_agent.eval.dataset` prints number questions from XBRL to copy in.
- Before marking a question `not_found`, search the filing to confirm the answer isn't there.
- The `correction` check only looks for a word, so read those answers yourself.
