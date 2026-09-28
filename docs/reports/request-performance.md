---
title: Where a request's time goes, and what was cut
description: A stage-by-stage profile of a Laya request on CPU, the speed-ups made from it with before and after figures, the int8 and ONNX fast paths, and a sizing of encoding the state once.
type: report
specificity: "Laya 0.3.20 on a 4-core Intel Xeon at 2.8 GHz (AVX-512 VNNI, no native BF16), PyTorch 2.14, ONNX Runtime 1.30, Python 3.11, fp32 on CPU with 4 threads. Checkpoints with the published architectures (ModernBERT-large, 421.29M parameters; mmBERT-base, 321.91M) and random weights."
credibility: "Measured with research/scripts/bench_stages.py, bench_question_heads.py, bench_cpu_fast_path.py and spike_encode_once.py; raw results are in research/results/*_2026092[56].json. Time depends on shapes, not weight values, so the split between stages holds for the real checkpoints. Answer agreement under int8 does not: random weights put nearly every answer on its boundary. No GPU or Apple-silicon runs."
---

# Where a request's time goes, and what was cut

## The profile

Milliseconds per request, median of 10 runs:

| stage | ticket, 1 question | ticket, 3 | ticket, 10 | email, 3 |
|---|---|---|---|---|
| routing (language detection) | 0.21 | 0.21 | 0.20 | 0.97 |
| tokenize the state | 0.34 | 0.35 | 0.34 | 1.18 |
| build rows (questions and options) | 0.41 | 1.02 | 2.59 | 0.97 |
| **encoder forward** | **260** | **505** | **1,390** | **2,239** |
| decision head forward | 18.5 | 27.1 | 80.2 | 167 |
| everything else | 0.63 | 0.93 | 1.40 | 1.13 |
| **latency** | **311** | **503** | **1,542** | **2,237** |

The ticket is 97 characters and the email 1,125. The encoder is 89-95% of every request
measured. Every question row carries the whole state, so time follows the number of questions
times the length of the state. The full table, with batches and Spanish, is in
[Where a request's time goes](../performance.md#the-stages).

## Speed-ups made, with identical output

| change | before | after |
|---|---|---|
| language detection, 2,000-character English text | 723 µs | 221 µs |
| language detection, JSON ticket | 1,610 µs | 946 µs |
| question heads tokenized once per tokenizer: 3 questions | 0.71 ms | 0.13 ms |
| question heads: 10 questions | 2.07 ms | 0.21 ms |
| question heads: batch of 32 tickets | 22.6 ms | 4.0 ms |
| routing a state seen before, 2,000 characters | 246 µs | 10 µs |
| a `DecisionCache` hit, 140 / 2,000 / 20,000 characters | 121 / 314 / 614 µs | 73 / 76 / 167 µs |

Language detection output was compared against the previous code: 267,449 comparisons over
38,247 inputs, none different. These cut milliseconds from a request that takes hundreds on CPU.
They matter more on an accelerator, where the encoder is short and the milliseconds around it
are a visible share.

## Runtime options tried

| option | ticket, 3 questions | email, 3 questions | answers |
|---|---|---|---|
| int8 dynamic quantization of the encoder | 1.80x faster | 1.64x faster | shifted |
| `torch.inference_mode()` | 1.02x | 1.01x | identical |
| 2 threads instead of 4 | 1.48x slower | 1.65x slower | identical |
| bf16 autocast without native BF16 | 3.10x slower | 3.47x slower | shifted |

Use every physical core, and use bf16 only on hardware with native support.

## The CPU fast path

Over 13 English requests (1 to 10 questions, tickets to 3,000-token emails), interleaved over 5
repeats:

| backend | ticket, 3 | email, 3 | geometric mean | same decision as fp32 |
|---|---|---|---|---|
| int8 PyTorch (encoder linear layers) | 1.85x | 1.68x | 1.89x | 44 of 56 |
| ONNX fp32, export before the fix | 1.04x | 0.79x | 1.01x | 56 of 56 |
| ONNX fp32, NaN guards stripped | 1.11x | 0.90x | 1.06x | 56 of 56 |
| **ONNX int8** | **1.96x** | **1.76x** | **2.07x** | 28 of 56 |

- **ONNX fp32 loses on long inputs** because the export unfuses attention into a full
  [heads, length, length] tensor per layer. It also added a NaN check after each softmax that
  cannot fire. `scripts/export_onnx.py` now strips those checks, which took the email from
  0.79x to 0.90x. Fusing the attention is the next step.
- **int8 changes answers.** Every changed answer had an fp32 margin of at most 0.0006, which is
  a coin flip under random weights. Trained checkpoints will change far fewer; measure yours.
  The int8 ONNX file is 579 MB, against 1.68 GB for fp32.
- **The cache pairs with int8.** Fp32, int8 and ONNX of one checkpoint share one set of
  decisions (see [the cache report](decision-cache-engineering.md#one-cache-for-every-backend)),
  so a repeated request gets its first answer whichever backend computed it.

## Trimming the state

Since time follows state length, trimming is the largest lever on a new request.
`laya.email.clean_email_body` drops quoted history, signatures and disclaimers, and caps the rest
at 3,000 characters:

| state | characters before | after |
|---|---|---|
| the benchmark email | 1,125 | 1,064 |
| a reply quoting it | 1,263 | 27 |
| a two-deep thread | 2,498 | 27 |

The quoted history is gone too, so keep it when the questions depend on it.

## Encoding the state once (research spike)

k questions put the state through the encoder k times. Encoding the state once and each
question separately, then letting the two head layers cross-attend, would cost this much of
today's forward pass (speedup, median of 3):

| state tokens | English, 1 question | 3 | 10 | multilingual, 3 | 10 |
|---|---|---|---|---|---|
| 100 | 0.60x | 1.45x | 2.32x | 1.19x | 2.06x |
| 300 | 0.83x | 2.07x | **4.18x** | 1.61x | 3.46x |
| 476 / 800 | 0.86x | 2.19x | **4.29x** | 2.09x | 3.70x |

It pays from about three questions on a hundred-token state, and loses on one question. It is
compute only: the answers mean nothing until a model is trained this way, most likely by
distilling from today's.

## Reproduce

```bash
python research/scripts/bench_stages.py --device cpu     # or mps, cuda
python research/scripts/bench_question_heads.py
python scripts/export_onnx.py --model ID --output english.onnx --int8
python research/scripts/bench_cpu_fast_path.py --onnx fp32=english.onnx --onnx int8=english.int8.onnx
python research/scripts/spike_encode_once.py
```
