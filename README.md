# Laya-Pro

**Typed decisions in one forward pass, with a memory for them.** Laya-Pro answers `choice`, `score`
and yes/no questions about any text, email, ticket or JSON document in a single forward pass, in
100+ languages, on your own hardware. It is built on the Laya decision engine and adds what
repeated production traffic needs: the same request gets the same decision every time, on any
machine and any backend, for as long as you choose to keep it, and a repeat costs microseconds
instead of a forward pass.

<p align="center">
  <img src="https://raw.githubusercontent.com/tgundhus/laya/main/assets/laya_pro_vs_laya_vs_jev.png" alt="Laya-Pro against Laya against TypeSafe Jev: the same request gives the same answer (Jev 22.7% identical across three runs, Laya 100% on one setup but 72-99.6% of decisions unchanged across precisions, Laya-Pro 100%); a repeated request takes 0.04-0.06 ms on Laya-Pro against 34-144 ms on Laya and 236-276 ms on Jev; accuracy where Jev numbers exist; 3.6 times the requests served at 70% repeats; decision memory by retention period; decisions changed when the setup changes" width="100%" />
</p>

## Laya-Pro and Laya

The models and the engine are Laya's, and without the decision cache installed, Laya-Pro's answers
are byte-identical to Laya's. The differences are in what happens around the model:

| | Laya | Laya-Pro |
|---|---|---|
| **The same request again** | runs the model again, and the decision can change with the device, precision or batch shape: up to 28% of decisions (int8) | returns the decision it gave the first time, on every device and backend: 841 of 841 replayed |
| **Memory of decisions** | none | a retention you choose per decision (`ttl`, renewed on use, a size bound), in memory, in SQLite or in your own store; survives restarts |
| **Cost of a repeat** | a forward pass: 34-491 ms | 0.04-0.06 ms |
| **Identical requests at the same moment** | one forward pass each | one forward pass, shared |
| **Borderline answers** | not reported | `decision_margins()` says how close each answer is to flipping |
| **Work around the model** | tokenizes each question and detects the language on every request | keeps question tokens and recent detections: 3.9-4.4x less work, the same answers |
| **ONNX export** | keeps attention NaN checks that cannot fire | strips them: 13% of an email's forward pass, the same answers |
| **A failing cache store** | - | never fails a prediction: the request is computed, and the failure counted |
| **Version** | 0.3.21 | 0.3.21 merged, with fixes to it |

## What Laya-Pro adds

### A memory for decisions

```python
from laya import DecisionCache, Router

cache = DecisionCache("decisions.sqlite", ttl=7 * 86400, renew_on_hit=True)
router = Router(hooks=[cache])

first = router.predict(state, questions)   # computed, then stored
again = router.predict(state, questions)   # replayed: the same decision, in microseconds
cache.cache_info()                          # size, hits, misses, coalesced, conflicts, errors
```

- **The same request.** The key is a 16-byte BLAKE2b hash of the state, the questions in their
  order, the token budget, the checkpoint and the model's fingerprint. No request text is stored.
- **How long it remembers.** Each decision carries its own expiry: `ttl` in seconds, `None` for
  never, or a function of the questions and the result. `renew_on_hit=True` keeps a decision alive
  while its request keeps coming, and `maxsize` bounds the store (100,000 decisions by default).
- **Where.** In memory, in a SQLite file that several processes share, or in your own
  `DecisionStore` ([a Redis example](examples/hooks/decision_store_redis.py) is included). The first
  decision stored wins, across threads, processes and machines.
- **Every backend, one decision.** fp32, int8 and ONNX share one key, on a `Router`, an `Agent` or
  an `ONNXAgent`, and identical requests in flight together share one forward pass.
- **Never in the way.** A store that fails, or holds a value it cannot decode, never fails a
  prediction: the request is computed, the first failure warns, and `cache_info()["errors"]` counts
  them all. A model upgrade changes the fingerprint, so old decisions are not replayed after it.

Guide: [Decision consistency](docs/consistency.md).

### How shaky an answer is

```python
from laya import decision_margins

margins = decision_margins(result)               # {"department": 0.62, "churn_risk": 0.03, ...}
review = [name for name, m in margins.items() if m < 0.05]
```

A margin is how far an answer is from flipping, from 0 on the boundary to 1. A margin under 0.05
flags 2.5% of answers and caught every decision that flipped between fp32 and fp16.

### Less work around the model

A question's instructions and options are tokenized once and reused across requests, the Router
remembers its recent language detections, and English detection takes ASCII shortcuts. The answers
are byte-identical, the work around the model is 3.9-4.4x smaller, and a new request is level on
CPU and up to 4% faster on an Apple GPU.

### A leaner ONNX export

`scripts/export_onnx.py` removes the NaN checks the exporter adds after every attention softmax,
which cannot fire (13% of an email's forward pass, with the same answers), and `--int8` (or
`--quantize`) writes a per-channel INT8 copy beside the fp32 model.

### Laya 0.3.21, merged and fixed

Everything Laya shipped up to 0.3.21 is merged: ONNX batch and long-document parity, per-channel
INT8, opt-in abstention with `min_confidence`, batch calls on every surface and per-request token
budgets. The cache works with all of it, and a review of the merge fixed:

- `ONNXAgent` now marks cached answers under `min_confidence`, and answers a long document from its
  cached windows instead of raising.
- `Router` keeps an `expected_sha256` pin passed in `agent_kwargs`. An empty per-model digest entry
  used to drop it and load the weights unverified.
- `Router.predict_batch` serves agents that predate `lang` and `sort_by_length`, LangChain's
  `abatch` accepts an empty batch, and the ONNX option-overflow error says how many options fit.

Details: [Merging Laya 0.3.21](docs/reports/real-checkpoints-m4-max.md#merging-laya-0321).

## Benchmarks

Laya-Pro against the original Laya (0.3.20, the version Laya-Pro branched from) on an Apple M4 Max,
with the published checkpoints, over 841 requests in 11 languages. Rows marked *cache* need
`DecisionCache` installed; the rest are the default path. Method and raw results: [Real checkpoints
on an Apple M4 Max](docs/reports/real-checkpoints-m4-max.md). The figure above comes from
[`make_laya_pro_plot.py`](research/scripts/make_laya_pro_plot.py), which reads the committed
results.

**Consistency**

| benchmark | Laya | Laya-Pro |
|---|---|---|
| Answers on 841 requests, CPU and Apple GPU (MPS) | reference | byte-identical to Laya |
| The same request alone and in a batch on MPS | 78% of payloads differ; 4 of 1,076 decisions flip | none differ (*cache*) |
| fp32 against fp16 or int8, on another device or backend | 0.4% (fp16) to 28% (int8) of decisions flip | none flip: 841 of 841 replayed (*cache*) |
| Spotting borderline answers | - | a margin under 0.05 flags 2.5% of answers and catches every fp16 flip |

**Memory of decisions** (*cache*)

| benchmark | Laya | Laya-Pro |
|---|---|---|
| Repeats given their earlier decision, a simulated month, 7-day retention | 0% | 89% (94% renewing on use) |
| The same with 30-day retention | 0% | 98% (99.8% renewing on use) |
| Decisions replayed by a fresh process after a restart (SQLite) | 0 | 20,000 of 20,000 |
| Retention rules kept exactly, over 348,004 checked requests | - | 0 disagreements |

**Speed**

| benchmark | Laya | Laya-Pro | change |
|---|---|---|---|
| Repeated request on MPS (3-question ticket / email) | 34.1 / 182 ms | 0.058 / 0.063 ms (*cache*) | 590-2,900x |
| Repeated request on CPU (the same requests) | 143.9 / 491 ms | 0.037 / 0.043 ms (*cache*) | 3,900-11,400x |
| New request, both versions around one model (3-question ticket) | 152.7 ms CPU, 49.0 ms MPS | 151.8 ms CPU, 48.2 ms MPS | level on CPU, up to 4% faster on MPS |
| Work around the model (3-question ticket / batch of 32) | 0.37 / 11.4 ms | 0.08 / 2.9 ms | 4.4x / 3.9x |
| Language detection, 2,000 characters (English / other languages) | 679 / 2,267 µs | 214 / 2,054 µs | 3.2x / 1.1x |
| A stream with 70% repeats on MPS | 16.8 requests/s, p99 144 ms | 61.0 requests/s, p99 93 ms (*cache*) | 3.6x |
| 16 identical new requests at once on CPU | 16 forward passes, 1.26 s | 1 forward pass, 0.22 s (*cache*) | 5.7x |

**Costs and limits**

| | measured |
|---|---|
| Memory held by the cache | 28-37 MB per 100,000 decisions, the default bound |
| SQLite hit and store | 4-7 µs and 21-29 µs median, with occasional 20-100 ms pauses for maintenance |
| int8 on Apple silicon | no faster than fp32 (ONNX) or 2.8x slower (PyTorch), and it changes 10-36% of decisions; PyTorch int8 lost 10.5 points of accuracy on MASSIVE English |
| Requests that differ only in format | reversed option order changed 19-36% of decisions, one trailing space up to 6.6%; the cache replays exact repeats only |

**After merging Laya 0.3.21**, against 0.3.21 as released, on the same machine:

| check | result |
|---|---|
| Answers on 841 requests | byte-identical on CPU and on MPS |
| Language detection, 179,214 inputs | identical, and 1.11-1.20x faster |
| Tokenizing the state and question rows | 1.4-6.8x faster on CPU |
| A new request around one model | 0.4-2.7% faster on MPS, level on CPU |
| A decision cache hit | 31 µs in memory, 36 µs on SQLite |

Other work shared the machine while these ran. The counts do not depend on timing, and the
new-request rows run both versions around one model in one process, in alternating pairs.

### Against Jev

Laya-Pro answers the same kind of typed questions as TypeSafe's Jev, a closed, hosted decision API,
and serves the same `/v1/systemone` protocol ([HTTP API](docs/http-api.md)). Where Jev falls short:

- **The same request does not get the same response.** Sent the same 64 requests three times, Jev
  1.13.0 returned identical responses for 29 of 128 (23%), and none of its 64 four-question yes/no
  answer sets repeated exactly: one label went `todo`, `todo`, then `noise`. Laya returned 128 of
  128 identical responses ([details](docs/reports/consistency-jev-vs-laya.md)), and Laya-Pro keeps
  that across devices and precisions.
- **Confident misses.** On DAIR Emotion, Jev gave the true label zero probability on 16% of
  examples.
- **Calibration.** ECE 0.246, against 0.081 after temperature fitting.
- **Latency, cost and privacy.** 236-276 ms p50 for one question at $0.042 per million tokens, and
  every request leaves your machine. Laya answers in 32.8 ms on a T4, self-hosted, and a Laya-Pro
  repeat takes 0.04-0.06 ms.

| accuracy | Jev 1.13.0 | Laya and Laya-Pro (routed) |
|---|---|---|
| typed-decisions, 2,000 decisions | 0.727 | **0.766** |
| AG News, 4 labels | 0.910 | **0.950** |
| DAIR Emotion, 6 labels | 0.480 | **0.595** |
| Banking77 (72 against 77 labels) | **0.870** | 0.425 |

Jev leads on large label sets, soft accuracy and raw calibration: [Where Jev
leads](docs/limits.md#where-jev-leads). The Jev figures are third-party published, since there is
no Jev API access here, except the repeat test, which reads archived raw responses in this
repository.

## Install

```bash
python -m pip install "laya @ git+https://github.com/tgundhus/laya.git"
```

Python 3.10 or newer. Extras go in brackets, as in
`"laya[serve,onnx] @ git+https://github.com/tgundhus/laya.git"`: `serve` (HTTP server), `mcp`,
`onnx`, `langchain`, `llamaindex`, `crewai`, `structured` and `fast` (the TileLang GPU path). The
package keeps the name `laya`, so `import laya` and the `laya` command work unchanged; `pip install
laya` from PyPI installs the original Laya, without Laya-Pro's changes. Platform notes:
[Installation](docs/guide.md#installation).

## Quickstart

```python
from laya import DecisionCache, Router, decision_margins

router = Router(hooks=[DecisionCache("decisions.sqlite", ttl=7 * 86400, renew_on_hit=True)])

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

result = router.predict(state, questions)
print(result["answers"]["department"]["choice"])  # billing
print(result["answers"]["churn_risk"]["noul"])    # the probability that the answer is yes
print(result["routing"]["model"])                 # english
print(decision_margins(result))                   # how close each answer is to flipping

router.predict(state, questions)                  # asked again: replayed from decisions.sqlite
```

The first call downloads the checkpoint. The Router detects the script and language, and sends
non-English text to the multilingual checkpoint. From the command line,
`laya "My payment failed twice" --preset triage` answers a ready-made question set.

## The engine

Laya evaluates typed questions over any state in a single forward pass: 33 ms for one question on a
T4, 7.2 ms per question batched. There is no text generation, so there is nothing to parse and
nothing to hallucinate.

| question | returns | for example |
|---|---|---|
| `choice` | the top label, a probability per option and a confidence | department, intent, topic |
| `score` | the expected level on an ordinal rubric, with its distribution | urgency, severity, frustration |
| `noul` | the calibrated probability that the answer is yes | phishing, spam, jailbreaks, churn risk |

Three checkpoints, and a `Router` that picks one per request:

| checkpoint | encoder | parameters | context | for |
|---|---|---|---|---|
| [`laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 421M | 512 | English |
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1,024 (up to 8,192) | 100+ languages, 2x faster |
| [`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1,024 | the typed-decisions workflows |

Routed accuracy, measured on a T4 by Laya's authors; Laya-Pro gives the same answers:

| benchmark | accuracy |
|---|---|
| MASSIVE intent, English, 20 options | 0.783 |
| MASSIVE intent, 13 other languages | 0.451 |
| XNLI, English / 14 other languages | 0.860 / 0.731 |
| Languages usable (over 3x random) | 45 of 51 |
| typed-decisions, the fine-tuned checkpoint | 0.766, against 0.362 for the base English one |

Also in the engine, each with its guide:

- **Routing:** script and language detection in under a millisecond, with `model=`, `lang=` and
  `lang_guess=` to decide it yourself. [Routing](docs/guide.md#routing)
- **Batches and long documents:** `predict_batch` shares forward passes, `predict_long` scans a
  document past the context window, and `laya-multilingual` reads 8,192 tokens with
  `max_len=8192`. [Batches](docs/guide.md#batches), [Long documents](docs/guide.md#long-documents)
- **Confidence and abstention:** calibrated probabilities, and `min_confidence` to flag answers
  below a threshold. [Confidence](docs/guide.md#confidence-and-abstention)
- **Schema-driven decisions:** `decide(state, schema=...)` from a JSON schema or a pydantic model.
  [Schema-driven decisions](docs/structured.md)
- **Serving:** `laya-serve` on the Jev-compatible `POST /v1/systemone`, Docker, an MCP server, the
  `laya` command, and `ONNXAgent` for ONNX Runtime. [HTTP API](docs/http-api.md),
  [Docker](docs/docker.md), [MCP](docs/guide.md#mcp-server), [command line](docs/guide.md#command-line)
- **Integrations:** [LangChain and LangGraph](docs/langchain.md), [LlamaIndex](docs/llamaindex.md)
  and [CrewAI](docs/crewai.md).
- **Hooks:** observe or change every decision; the decision cache is one. [Hooks](docs/hooks/index.md)
- **Fine-tuning:** where the accuracy jumps, on Kaggle's free GPUs. [Fine-tuning](docs/finetune.md)

## Limits

- The base checkpoints are near chance on typed-decisions zero-shot (0.362, against a 0.318 random
  baseline). Fine-tune for your own decisions.
- Choice questions with more than about 20 options lose accuracy (Banking77: 0.425). Raise
  `head_max_len` for that request, or shortlist the options with `predict_shortlist`.
- The shipped checkpoints are over-confident, and `laya-multilingual` has no fitted temperatures.
  Calibrate before you trust a probability.
- `score` is the weakest question type, and on the English checkpoint `noul` can follow its option
  labels instead of the text; give `noul` questions criteria.
- The cache replays exact repeats only, and `laya-serve` and the MCP server do not install it yet.

All of them, measured: [Limits](docs/limits.md).

## Documentation

- [Using Laya-Pro](docs/guide.md): installation, routing, batches, long documents, confidence and
  the servers
- [Decision consistency](docs/consistency.md): the decision cache and decision margins
- [Where a request's time goes](docs/performance.md)
- [Limits](docs/limits.md)
- [Reports](docs/reports/index.md), and [every benchmark run](BENCHMARKS.md)
- [Python API reference](docs/reference/index.md)

## Credits and license

Laya-Pro is built on **Laya**, created by Nandakishor M and developed by Convai Innovations. The
models, the RLCD training method, the checkpoints and the engine are theirs, published under the
Apache License 2.0. Laya-Pro's additions (the decision cache, decision margins, the work around the
model, the ONNX export changes and the benchmarks here) are by Tobias Gundhus, under the same
license. Laya-Pro is an independent fork, not affiliated with or endorsed by Convai Innovations.
See [LICENSE](LICENSE).
