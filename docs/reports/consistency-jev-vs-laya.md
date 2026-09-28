---
title: "Consistency: Jev against Laya"
description: How often Jev 1.13.0 and Laya return the same response to the same request, what the variation looks like, and whether voting or replaying removes it.
type: report
specificity: "The archived Feishu run in research/benchmarks/feishu_zh: 64 synthetic Chinese workplace cases, each sent 3 times to Jev 1.13.0 over its API and to laya-multilingual (Laya 0.3.20) on an Apple M4 in float32. 128 responses per system: 64 choice-mode and 64 four-question noul-mode."
credibility: "Measured from archived raw responses. tests/test_consistency.py recomputes every figure on this page from them, and fails if any changes. The voting formula is exact for independent runs; Jev's runs may not be independent, so treat it as a best case for voting."
---

# Consistency: Jev against Laya

An article on Jev reported that only 24% of its responses were identical when a request was
repeated. The archived Feishu comparison checks that figure and puts Laya beside it.

## Identical responses across three runs

| identical across three runs | Jev 1.13.0 | Laya |
|---|---|---|
| whole response, probabilities and `confidence` included | 29 of 128 (23%) | 128 of 128 |
| `choice` probabilities | 38 of 64 | 64 of 64 |
| yes/no values of the four-question mode | 0 of 64 | 64 of 64 |
| the decision the benchmark acts on (the label) | 127 of 128 | 128 of 128 |

The 23% here matches the article's 24%. Laya returned the same bytes every time, with no cache
installed. Its forward pass samples nothing, so on a fixed setup there is nothing to vary.

## What Jev's variation looks like

- Yes/no values moved by a median of 0.01, and at most 0.08.
- In 9 answers, `confidence` changed while the probabilities it printed, to two decimals, did
  not.
- One label changed: `cross_chat-03` came back as `todo`, `todo` and `noise`. It was the least
  decided answer in the set: 0.53 against 0.44, then 0.51 against 0.44, then 0.45 against 0.50.

Most of the variation is in the digits, not the decision, but a digit is enough to flip an
answer that sits on its boundary.

## Margins find the answers that flip

`decision_margins(result)` returns, per question, how far the answer is from flipping, from 0
to 1. On the archive:

- a margin under 0.1 marked all three of Jev's changed answers, in every run;
- it marked 26 of the 960 answers Jev gave across the runs (2.7%).

For Laya, whose answers do not move between runs, the margin estimates how likely an answer is
to be wrong. On the base multilingual checkpoint, 1 of the 14 `choice` answers with a margin
under 0.05 was right, against 14 of the 36 with a margin of 0.2 or more.

## What can still move a Laya answer

On a fixed setup, nothing. Across setups:

- hardware and precision: CUDA bf16 or fp16 autocast, and MPS fp16 from five rows up;
- batch shape: padding changes floating-point summation order;
- the model: a new revision, a refitted temperature or a `max_len` change;
- the request: option order is positional, and on 20 options the answer changed for 15-23% of
  permutations.

Each of these only flips an answer close to its boundary.

## Voting against replaying

| approach | what it does to run-to-run variation | cost |
|---|---|---|
| majority of three runs | a flip rate q becomes 3q² − 2q³, which is 26% at q = 1/3 | three calls per request |
| averaging the probabilities of the runs | better than counting votes; still one draw of an average | the same three calls |
| re-asking only when the margin is under 0.1 | the same, on 2.7% of answers | 2.7% extra calls on the archive |
| **replaying the first decision (`DecisionCache`)** | **removes it for as long as the entry lives** | **73 µs per repeat** |

Voting over repeats is pointless for Laya, whose repeats are identical ballots. A vote over
*equivalent* requests is a different tool. Averaging over the rotations of a `choice`
question's options cancels its option-position bias; the recipe is in
[Decision consistency](../consistency.md#voting).

Replaying makes repeats identical; it does not make a borderline answer right, and it keeps a
wrong one until the entry expires. Use the margin to decide which answers need review.

## Reproduce

```bash
python tests/test_consistency.py   # recomputes every figure above from the archive
```
