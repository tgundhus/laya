"""Decision consistency: `decision_margins` and the `DecisionCache` hook, without a model.

The Router path attaches counting fakes. The Agent path runs the real `predict_batch` with its
tokenizer and forward pass stubbed, as `test_hooks.py` does, so the cache is exercised against
the engine's own hook sequencing. The last section replays the archived Feishu runs to check
that a 0.1 margin flags every answer that changed between Jev's repeated runs.

Run: python tests/test_consistency.py
"""
import json
import os
import sys
import tempfile
import threading
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import laya  # noqa: E402
import laya.consistency as consistency  # noqa: E402
from laya import DecisionCache, Router, decision_margins  # noqa: E402
from laya.agent import Agent  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
        return
    except BaseException as other:  # noqa: BLE001
        FAIL.append("%s: raised %r, want %r" % (name, other, exc))
        return
    FAIL.append("%s: did not raise %r" % (name, exc))


class Clock:
    """Stands in for `laya.consistency._now` so expiry is tested without sleeping."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


clock = Clock()
consistency._now = clock

# --------------------------------------------------------------- decision_margins
choice = {"type": "choice", "choice": "todo", "probabilities": {"todo": 0.53, "noise": 0.44, "urgent": 0.03}}
check("margins/choice is the top-two gap", decision_margins({"answers": {"c": choice}}), {"c": 0.09})
check("margins/an answers mapping works too", decision_margins({"c": choice}), {"c": 0.09})
check("margins/one option cannot flip",
      decision_margins({"c": {"type": "choice", "probabilities": {"only": 1.0}}}), {"c": 1.0})
check("margins/score uses the level probabilities",
      decision_margins({"s": {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}}),
      {"s": 0.3})
check("margins/noul at the default threshold",
      decision_margins({"a": {"noul": 0.54}, "b": {"noul": 0.5}, "c": {"noul": 1.0}, "d": {"noul": 0.0}}),
      {"a": 0.08, "b": 0.0, "c": 1.0, "d": 1.0})
check("margins/noul equals the P(yes) - P(no) gap at 0.5",
      decision_margins({"n": {"noul": 0.8}})["n"],
      decision_margins({"c": {"probabilities": {"yes": 0.8, "no": 0.2}}})["c"])
check("margins/per-question thresholds, missing names use 0.5",
      decision_margins({"urgent": {"noul": 0.8}, "related": {"noul": 0.6}}, noul_threshold={"urgent": 0.75}),
      {"urgent": 0.1, "related": 0.2})
check("margins/one threshold for every noul", decision_margins({"n": {"noul": 0.8}}, noul_threshold=0.75),
      {"n": 0.1})
check("margins/entries without numbers are left out",
      decision_margins({"a": {"noul": None}, "b": "text", "c": {"probabilities": {}}, "d": {"choice": "x"}}), {})
check_raises("margins/threshold outside [0, 1]", ValueError, lambda: decision_margins({"n": {"noul": 0.5}}, 1.5))
check_raises("margins/per-question threshold outside [0, 1]", ValueError,
             lambda: decision_margins({"n": {"noul": 0.5}}, {"n": -0.1}))
check_raises("margins/not a mapping", TypeError, lambda: decision_margins(["a"]))
jev = {"model": "jev-1.13.0", "answers": {"related": {"type": "noul", "noul": 0.86},
                                          "category": {"type": "choice", "choice": "todo", "confidence": 0.37,
                                                       "probabilities": {"todo": 0.53, "noise": 0.44}}}}
check("margins/reads a Jev response", decision_margins(jev), {"related": 0.72, "category": 0.09})


# --------------------------------------------------------------- Router fakes
class CountingAgent:
    """Answers every noul question with `answer` and records each forward pass."""

    model_id = "fake/checkpoint"
    revision = "r1"

    def __init__(self, answer=0.7):
        self.answer = answer
        self.calls = []
        self.temperature = [1.0, 1.0, 1.0]
        self.cfg = {"max_len": 512, "head_max_len": 192}

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        self.calls.append((list(states), overrides))
        return [{"model": "laya-rl-agent",
                 "answers": {q: {"type": "noul", "noul": self.answer} for q in questions},
                 "usage": {"input_tokens": len(str(s)), "output_tokens": 0}} for s in states]

    def system_one(self, state, questions, **overrides):
        return self.predict_batch([state], questions, **overrides)[0]


def router_with(cache, english=None, multilingual=None):
    r = Router(hooks=[cache])
    en = english or CountingAgent()
    ml = multilingual or CountingAgent(0.2)
    r.attach("english", en)
    r.attach("multilingual", ml)
    return r, en, ml


Q = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
STATE = "Hi, we were billed twice for March. Please refund the duplicate today."

# --------------------------------------------------------------- hits, misses, replays
cache = DecisionCache()
r, en, ml = router_with(cache)
first = r.predict(STATE, Q)
second = r.predict(STATE, Q)
check("cache/a repeat skips the forward pass", len(en.calls), 1)
check("cache/the replay is the stored answer", second["answers"], first["answers"])
check("cache/the replay keeps usage", second["usage"], first["usage"])
check("cache/the Router adds routing to a replay", second["routing"]["model"], "english")
check("cache/counters", cache.cache_info(),
      {"size": 1, "maxsize": 100_000, "hits": 1, "misses": 1, "conflicts": 0, "coalesced": 0, "errors": 0})

first["answers"]["billing"]["noul"] = -1.0
second["answers"]["billing"]["noul"] = -2.0
check("cache/a caller mutating a result does not change the store",
      r.predict(STATE, Q)["answers"]["billing"]["noul"], 0.7)

r.predict(STATE + " ", Q)
check("cache/a different state is a different request", len(en.calls), 2)
swapped = {"c": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}}
reordered = {"c": {"type": "choice", "instructions": "?", "criteria": {"b": "y", "a": "x"}}}
r.predict(STATE, swapped)
r.predict(STATE, reordered)
check("cache/option order is part of the key (#166)", len(en.calls), 4)
r.predict(STATE, Q, max_len=128)
check("cache/a per-call token budget is part of the key", len(en.calls), 5)
r.predict(STATE, Q, model="multilingual")
check("cache/the routed checkpoint is part of the key", len(ml.calls), 1)
r.predict(STATE, Q, lang="de")
check("cache/without per-language temperatures lang cannot change the answer", len(ml.calls), 1)
before = len(en.calls)
r.predict(STATE, {})
check("cache/an empty question set is left to the engine", cache.cache_info()["size"], 6)
check("cache/nothing inferred for empty questions here", len(en.calls), before + 1)


class LangAgent(CountingAgent):
    def __init__(self):
        super().__init__(0.2)
        self.lang_temperatures = {"de": {"temperature": [1.0, 1.0, 2.0], "temperature_by_options": {}}}


cache = DecisionCache()
r, _, ml = router_with(cache, multilingual=LangAgent())
r.predict(STATE, Q, lang="de")
r.predict(STATE, Q, lang="fr")
check("cache/with per-language temperatures an explicit lang is part of the key", len(ml.calls), 2)
r.predict(STATE, Q, lang="de")
check("cache/the same explicit lang replays", len(ml.calls), 2)
r.predict("मुझसे मार्च में दो बार शुल्क लिया गया", Q)
r.predict("मुझसे मार्च में दो बार शुल्क लिया गया", Q)
check("cache/a detected language replays too", len(ml.calls), 3)

# --------------------------------------------------------------- fingerprint
cache = DecisionCache()
r, en, _ = router_with(cache)
r.predict(STATE, Q)
en.temperature = [1.0, 1.0, 2.0]
r.predict(STATE, Q)
check("fingerprint/new calibration starts fresh decisions", len(en.calls), 2)
en.revision = "r2"
r.predict(STATE, Q)
check("fingerprint/new revision starts fresh decisions", len(en.calls), 3)
pinned = DecisionCache(fingerprint="policy-v1")
r, en, _ = router_with(pinned)
r.predict(STATE, Q)
en.temperature = [9.0, 9.0, 9.0]
en.revision = "r9"
r.predict(STATE, Q)
check("fingerprint/a fixed fingerprint replays across upgrades", len(en.calls), 1)

# --------------------------------------------------------------- expiry and size
cache = DecisionCache(ttl=60)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
clock.t += 59
r.predict(STATE, Q)
check("ttl/replayed while younger than ttl", len(en.calls), 1)
clock.t += 1
r.predict(STATE, Q)
check("ttl/recomputed once ttl old", len(en.calls), 2)
check("ttl/the recomputed decision replaces the old one", cache.cache_info()["size"], 1)

cache = DecisionCache(maxsize=2)
r, en, _ = router_with(cache)
for s in ("one", "two", "three"):
    r.predict(s, Q)
check("maxsize/bounded", cache.cache_info()["size"], 2)
r.predict("one", Q)
check("maxsize/the oldest decision left first", len(en.calls), 4)
r.predict("three", Q)
check("maxsize/newer decisions stay", len(en.calls), 4)

cache = DecisionCache(ttl=10)
r, en, _ = router_with(cache)
r.predict("a", Q)
clock.t += 5
r.predict("b", Q)
clock.t += 6
check("prune/drops what is expired", cache.prune(), 1)
check("prune/keeps the rest", cache.cache_info()["size"], 1)
cache.cache_clear()
check("clear/empties and resets", cache.cache_info(),
      {"size": 0, "maxsize": 100_000, "hits": 0, "misses": 0, "conflicts": 0, "coalesced": 0, "errors": 0})

check_raises("args/ttl must be positive", ValueError, lambda: DecisionCache(ttl=0))
check_raises("args/ttl is not a bool", ValueError, lambda: DecisionCache(ttl=True))
check_raises("args/maxsize must be positive", ValueError, lambda: DecisionCache(maxsize=0))
check_raises("args/maxsize is an int", ValueError, lambda: DecisionCache(maxsize=2.5))
check_raises("args/fingerprint is a string", TypeError, lambda: DecisionCache(fingerprint=1))
check("args/no bound and no expiry are allowed",
      DecisionCache(ttl=None, maxsize=None).cache_info()["maxsize"], None)


# --------------------------------------------------------------- results that are not JSON
class SetAgent(CountingAgent):
    def predict_batch(self, states, questions, batch_size=None, **overrides):
        results = super().predict_batch(states, questions, batch_size, **overrides)
        for result in results:
            result["extra"] = {1, 2}
        return results


cache = DecisionCache()
r, en, _ = router_with(cache, english=SetAgent())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = r.predict(STATE, Q)
    r.predict(STATE, Q)
check("json/the caller still gets the result", out["extra"], {1, 2})
check("json/a result that cannot be replayed exactly is not cached", len(en.calls), 2)
check("json/warned once", sum("not plain JSON" in str(w.message) for w in caught), 1)

# --------------------------------------------------------------- Router.predict_batch
cache = DecisionCache()
r, en, _ = router_with(cache)
r.predict("a", Q)
out = r.predict_batch([{"state": s, "questions": Q, "model": "english"} for s in ("a", "b", "b", "c")])
check("batch/only misses reach the model", en.calls[-1][0], ["b", "b", "c"])
check("batch/results stay in request order", [o["usage"]["input_tokens"] for o in out], [1, 1, 1, 1])
check("batch/every result is routed", [o["routing"]["model"] for o in out], ["english"] * 4)
check("batch/duplicates in one batch store one decision", cache.cache_info()["size"], 3)
r.predict_batch([{"state": s, "questions": Q, "model": "english"} for s in ("a", "b", "c")])
check("batch/a repeated batch is served from the cache", len(en.calls), 2)
check("batch/predict and predict_batch share entries", r.predict("c", Q)["answers"], out[3]["answers"])
check("batch/still no extra forward pass", len(en.calls), 2)


# --------------------------------------------------------------- first decision wins
class Racing(CountingAgent):
    """While computing its own answer, lets a sibling process store a different one first."""

    def __init__(self, answer, sibling):
        super().__init__(answer)
        self.sibling = sibling

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        self.sibling.predict(states[0], questions)
        return super().predict_batch(states, questions, batch_size, **overrides)


with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "decisions.sqlite")
    cache_a, cache_b = DecisionCache(path), DecisionCache(path)
    r_a, _, _ = router_with(cache_a, english=CountingAgent(0.51))
    r_b, en_b, _ = router_with(cache_b, english=Racing(0.49, r_a))
    out = r_b.predict(STATE, Q)
    check("race/the request that stored first decides for both", out["answers"]["billing"]["noul"], 0.51)
    check("race/the replaced decision is counted", cache_b.cache_info()["conflicts"], 1)
    check("race/the replacement keeps its routing", out["routing"]["model"], "english")
    check("race/one decision stored", cache_b.cache_info()["size"], 1)
    check("race/later requests replay it", r_b.predict(STATE, Q)["answers"]["billing"]["noul"], 0.51)
    check("race/without another pass", len(en_b.calls), 1)
    cache_a.close()
    cache_b.close()

# --------------------------------------------------------------- concurrent identical requests
class Slow(CountingAgent):
    """Takes `delay` seconds per forward pass, so concurrent callers overlap it."""

    def __init__(self, answer=0.7, delay=0.3):
        super().__init__(answer)
        self.delay = delay
        self.mutex = threading.Lock()

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        time.sleep(self.delay)
        with self.mutex:
            return super().predict_batch(states, questions, batch_size, **overrides)


def concurrently(n, call):
    """Run `call()` on n threads released together; return their results in thread order."""
    barrier, out = threading.Barrier(n), [None] * n

    def run(i):
        barrier.wait()
        out[i] = call()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return out


cache = DecisionCache()
r, slow, _ = router_with(cache, english=Slow())
outs = concurrently(8, lambda: r.predict(STATE, Q))
info = cache.cache_info()
check("coalesce/eight concurrent identical requests run one forward pass", len(slow.calls), 1)
check("coalesce/every caller gets the one decision", len({json.dumps(o["answers"]) for o in outs}), 1)
check("coalesce/every caller gets its routing", [o["routing"]["model"] for o in outs], ["english"] * 8)
check("coalesce/the others are hits", (info["hits"], info["misses"]), (7, 1))
check_true("coalesce/the waits are counted", info["coalesced"] >= 1, str(info))
check("coalesce/no claim outlives its request", cache._inflight, {})
check("coalesce/nothing conflicts", info["conflicts"], 0)

cache = DecisionCache(coalesce=False)
r, slow, _ = router_with(cache, english=Slow())
outs = concurrently(4, lambda: r.predict(STATE, Q))
check_true("coalesce/off: each request computes", len(slow.calls) > 1, str(len(slow.calls)))
check("coalesce/off: the first decision still wins", len({json.dumps(o["answers"]) for o in outs}), 1)

# different requests never wait for each other
cache = DecisionCache()
r, slow, _ = router_with(cache, english=Slow(delay=0.2))
t0 = time.perf_counter()
concurrently(4, lambda: r.predict("state %s" % threading.get_ident(), Q))
check("coalesce/distinct requests all compute", len(slow.calls), 4)
check_true("coalesce/distinct requests are not serialised", time.perf_counter() - t0 < 0.7,
           "%.2fs" % (time.perf_counter() - t0))

# a batch holding the same request twice runs it once and does not wait on itself
cache = DecisionCache()
r, en, _ = router_with(cache)
out = r.predict_batch([{"state": "dup", "questions": Q, "model": "english"}] * 3)
check("coalesce/a batch never waits on its own requests", len(out), 3)
check("coalesce/nothing claimed after the batch", cache._inflight, {})

# a failed forward pass releases its waiters, which then compute for themselves
class Failing(Slow):
    def __init__(self):
        super().__init__(delay=0.2)
        self.failed = False

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        with self.mutex:
            first, self.failed = not self.failed, True
        if first:
            time.sleep(self.delay)
            raise RuntimeError("out of memory")
        return super().predict_batch(states, questions, batch_size, **overrides)


def attempt():
    try:
        return r.predict(STATE, Q)
    except RuntimeError as exc:
        return exc


cache = DecisionCache()
r, failing, _ = router_with(cache, english=Failing())
t0 = time.perf_counter()
outs = concurrently(4, attempt)
check("coalesce/one caller sees the failure", sum(isinstance(o, RuntimeError) for o in outs), 1)
check("coalesce/the rest still get a decision", sum(isinstance(o, dict) for o in outs), 3)
check_true("coalesce/waiters are released at once, not at the wait bound", time.perf_counter() - t0 < 5,
           "%.2fs" % (time.perf_counter() - t0))
check("coalesce/the failure leaves no claim", cache._inflight, {})

# hooks that could be waited on in return never wait
serial = DecisionCache()
r, slow, _ = router_with(serial, english=Slow(delay=0.2))
r.hooks_concurrent = False
r._hooks_lock = threading.Lock()
outs = concurrently(3, lambda: r.predict(STATE, Q))
check("coalesce/under a hooks lock nothing is claimed", serial._inflight, {})
check("coalesce/under a hooks lock every caller answers", sum(isinstance(o, dict) for o in outs), 3)
timed = DecisionCache()
r, slow, _ = router_with(timed, english=Slow(delay=0.2))
r.hooks_timeout = 10
outs = concurrently(3, lambda: r.predict(STATE, Q))
check("coalesce/under hooks_timeout nothing waits", timed.cache_info()["coalesced"], 0)
check("coalesce/under hooks_timeout every caller answers", sum(isinstance(o, dict) for o in outs), 3)

# a stuck owner holds its waiters only until the wait bound, then its claim is dropped
cache = DecisionCache()
stuck = threading.Event()
cache._inflight[b"k" * 16] = (-1, stuck)
saved, consistency._COALESCE_WAIT = consistency._COALESCE_WAIT, 0.05
try:
    filled, owned = cache._claim([b"k" * 16], [None])
finally:
    consistency._COALESCE_WAIT = saved
check("coalesce/a stuck owner's claim is taken over", [key for key, _ in owned], [b"k" * 16])
check("coalesce/and nothing was filled", filled, 0)
cache._release(owned)
check("coalesce/the take-over releases too", cache._inflight, {})

# --------------------------------------------------------------- one cache for fp32, int8 and ONNX
class Int8(CountingAgent):
    """The same checkpoint quantized: another class, another dtype, the same fingerprint inputs."""
    dtype = "qint8"
    amp_enabled = False


class OnnxLike(CountingAgent):
    backend = "onnxruntime"
    precision = "int8"


cache = DecisionCache()
keys = [cache._keys(laya.PredictContext(states=[STATE], questions=Q, model="english", agent=agent_cls()),
                    [STATE], False)[0] for agent_cls in (CountingAgent, Int8, OnnxLike)]
check("backends/fp32, int8 and ONNX of one checkpoint share a key", len(set(keys)), 1)
r_fp32, fp32, _ = router_with(cache, english=CountingAgent(0.7))
r_int8, int8, _ = router_with(cache, english=Int8(0.69))
r_onnx, onnx, _ = router_with(cache, english=OnnxLike(0.71))
answers = [rt.predict(STATE, Q)["answers"]["billing"]["noul"] for rt in (r_fp32, r_int8, r_onnx)]
check("backends/int8 and ONNX replay the fp32 decision", answers, [0.7, 0.7, 0.7])
check("backends/only the first backend ran", (len(fp32.calls), len(int8.calls), len(onnx.calls)), (1, 0, 0))

# --------------------------------------------------------------- SQLite store
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "decisions.sqlite")
    cache = DecisionCache(path, ttl=100)
    r, en, _ = router_with(cache)
    stored = r.predict(STATE, Q)
    cache.close()
    reopened = DecisionCache(path, ttl=100)
    r, en, _ = router_with(reopened)
    again = r.predict(STATE, Q)
    check("sqlite/decisions survive a restart", len(en.calls), 0)
    check("sqlite/the replay is the stored answer", again["answers"], stored["answers"])
    clock.t += 100
    r.predict(STATE, Q)
    check("sqlite/expired decisions are recomputed", len(en.calls), 1)
    clock.t += 50
    r.predict("fresh", Q)
    clock.t += 60
    check("sqlite/prune drops what is expired", reopened.prune(), 1)
    check("sqlite/prune keeps the rest", reopened.cache_info()["size"], 1)
    reopened.cache_clear()
    check("sqlite/clear", reopened.cache_info()["size"], 0)
    reopened.close()

    bounded = DecisionCache(os.path.join(tmp, "bounded.sqlite"), maxsize=3)
    r, en, _ = router_with(bounded)
    for i in range(10):
        clock.t += 1
        r.predict("state %d" % i, Q)
    check("sqlite/maxsize bound", bounded.cache_info()["size"], 3)
    r.predict("state 9", Q)
    r.predict("state 0", Q)
    check("sqlite/the oldest decisions left first", len(en.calls), 11)
    bounded.close()


# --------------------------------------------------------------- Agent path (real predict_batch)
def make_fake():
    """A real `predict_batch` whose tokenizer and forward pass are stubbed; answers name the state."""
    fake = Agent.__new__(Agent)
    fake.tok = type("Tok", (), {"pad_token_id": 0})()
    fake._to_internal = staticmethod(Agent._to_internal).__func__
    fake.encoded = []

    def _encode_state(state, ids, internal, **overrides):
        fake.encoded.append(state)
        return [{"ids": [1, 2, 3], "markers": [0, 1], "qtype": 2, "state": state} for _ in ids]

    def _forward(b):
        n = b["input_ids"].shape[0]
        return np.zeros((n, 2), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode_answers(logits, act, items, ids, internal, offset, **kw):
        return {qid: {"type": "noul", "noul": 0.5, "of": items[j]["state"]} for j, qid in enumerate(ids)}

    fake._encode_state = _encode_state
    fake._forward = _forward
    fake._decode_answers = _decode_answers
    return fake


seen_states = []
cache = DecisionCache()
agent = make_fake()
agent.hooks = [cache]
first = agent.predict_batch(["s0", "s1"], Q)
out = agent.predict_batch(["s0", "s2", "s1"], Q, on_predict_end=lambda c: seen_states.append(list(c.states)))
check("agent/only the missing state is tokenized", agent.encoded, ["s0", "s1", "s2"])
check("agent/results line up with the states", [o["answers"]["billing"]["of"] for o in out], ["s0", "s2", "s1"])
check("agent/replays equal the first answers", [out[0], out[2]], first)
check("agent/later end hooks see every state", seen_states, [["s0", "s2", "s1"]])
check("agent/usage covers every state", [o["usage"]["input_tokens"] for o in out], [3, 3, 3])
agent.encoded.clear()
again = agent.predict_batch(("s1", "s2"), Q)
check("agent/an all-hit batch skips inference", agent.encoded, [])
check("agent/an all-hit batch returns every replay", [o["answers"]["billing"]["of"] for o in again], ["s1", "s2"])
check("agent/system_one hits too", agent.system_one("s0", Q)["answers"], first[0]["answers"])
check("agent/still nothing tokenized", agent.encoded, [])

per_lang = make_fake()
per_lang.hooks = [DecisionCache()]
per_lang.lang_temperatures = {"de": {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}}
per_lang.predict_batch(["s0"], Q, lang="de")
per_lang.predict_batch(["s0"], Q, lang="fr")
check("agent/with per-language temperatures the cache stands aside", per_lang.encoded, ["s0", "s0"])

# --------------------------------------------------------------- retention policy
DAY = 24 * 3600
FRAUD = {"fraud": {"type": "noul", "instructions": "Is this fraud?"}}


def by_question_set(questions, result):
    return 7 * DAY if "fraud" in questions else 30 * DAY


cache = DecisionCache(ttl=by_question_set)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
r.predict(STATE, FRAUD)
clock.t += 8 * DAY
r.predict(STATE, Q)
r.predict(STATE, FRAUD)
check("policy/ttl per question set: the 30-day set replays, the 7-day one expired",
      [c[0] for c in en.calls], [[STATE], [STATE], [STATE]])


def keep_close_calls_longer(questions, result):
    margins = decision_margins(result)
    return 90 * DAY if min(margins.values(), default=1.0) < 0.1 else DAY


cache = DecisionCache(ttl=keep_close_calls_longer)
close = CountingAgent(0.52)
r, _, _ = router_with(cache, english=close)
r.predict(STATE, Q)
clock.t += 2 * DAY
r.predict(STATE, Q)
check("policy/ttl by margin: a close call is kept longer", len(close.calls), 1)
clear = CountingAgent(0.95)
r, _, _ = router_with(DecisionCache(ttl=keep_close_calls_longer), english=clear)
r.predict(STATE, Q)
clock.t += 2 * DAY
r.predict(STATE, Q)
check("policy/ttl by margin: a clear answer expires after a day", len(clear.calls), 2)

seen = []
cache = DecisionCache(ttl=lambda questions, result: seen.append(sorted(result)) or 0)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
r.predict(STATE, Q)
check("policy/a lifetime of 0 keeps nothing", (len(en.calls), cache.cache_info()["size"]), (2, 0))
check("policy/the function sees the payload without routing", seen[0], ["answers", "model", "usage"])

forever = DecisionCache(ttl=lambda questions, result: None)
r, en, _ = router_with(forever)
r.predict(STATE, Q)
clock.t += 10 ** 6
r.predict(STATE, Q)
check("policy/None keeps a decision for ever", len(en.calls), 1)

broken = DecisionCache(ttl=lambda questions, result: "a week")
r, _, _ = router_with(broken)
check_raises("policy/a lifetime that is not a number raises", TypeError, lambda: r.predict(STATE, Q))
check_raises("args/ttl is a positive number or a function", ValueError, lambda: DecisionCache(ttl="1d"))

# --------------------------------------------------------------- retention from last use
for label, make in (("memory", lambda tmp: DecisionCache(ttl=100, renew_on_hit=True)),
                    ("sqlite", lambda tmp: DecisionCache(os.path.join(tmp, "renew.sqlite"), ttl=100,
                                                         renew_on_hit=True))):
    with tempfile.TemporaryDirectory() as tmp:
        cache = make(tmp)
        r, en, _ = router_with(cache)
        r.predict(STATE, Q)
        for _ in range(5):
            clock.t += 90
            r.predict(STATE, Q)
        check("renew/%s: a decision in use outlives its ttl" % label, len(en.calls), 1)
        clock.t += 101
        r.predict(STATE, Q)
        check("renew/%s: it expires ttl after its last use" % label, len(en.calls), 2)
        cache.close()


class SpyStore:
    """A dict-backed DecisionStore that records calls: the protocol is all a store needs."""

    def __init__(self):
        self.data, self.calls = {}, []

    def get(self, key, now):
        self.calls.append("get")
        entry = self.data.get(key)
        return entry if entry is not None and entry[1] > now else None

    def add(self, items, now):
        self.calls.append("add")
        held = []
        for key, value, expires_at in items:
            if self.get(key, now) is None:
                self.data[key] = (value, expires_at)
            held.append(self.data[key][0])
        return held

    def renew(self, key, expires_at, now):
        self.calls.append("renew")
        if key in self.data:
            self.data[key] = (self.data[key][0], expires_at)

    def prune(self, now):
        expired = [k for k, (_, e) in self.data.items() if e <= now]
        for k in expired:
            del self.data[k]
        return len(expired)

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def close(self):
        self.calls.append("close")


spy = SpyStore()
cache = DecisionCache(store=spy, ttl=160, renew_on_hit=True)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
for _ in range(40):
    clock.t += 1
    r.predict(STATE, Q)
check("store/a custom store serves hits", len(en.calls), 1)
# ttl 160: renewed once more than 10 s (a 16th) of it has passed -- at 11, 22 and 33 s of 40 hits
check("store/renewals are written once a decision has aged a 16th of its ttl", spy.calls.count("renew"), 3)
check("store/cache_info reports no maxsize for a custom store", cache.cache_info()["maxsize"], None)
check("store/size comes from the store", cache.cache_info()["size"], 1)
clock.t += 200
check("store/prune goes to the store", cache.prune(), 1)
cache.close()
check("store/close goes to the store", spy.calls[-1], "close")

first = SpyStore()
cache_a, cache_b = DecisionCache(store=first), DecisionCache(store=first)
r_a, _, _ = router_with(cache_a, english=CountingAgent(0.51))
r_b, _, _ = router_with(cache_b, english=Racing(0.49, r_a))
check("store/first decision wins through a shared custom store",
      r_b.predict(STATE, Q)["answers"]["billing"]["noul"], 0.51)
check_raises("store/path and store together", ValueError, lambda: DecisionCache("x.sqlite", store=SpyStore()))
check_raises("store/maxsize with a custom store", ValueError, lambda: DecisionCache(store=SpyStore(), maxsize=5))
check_raises("store/an object without the protocol", TypeError, lambda: DecisionCache(store=object()))
check("store/maxsize=None with a custom store says the store bounds itself",
      DecisionCache(store=SpyStore(), maxsize=None).maxsize, None)


# --------------------------------------------------------------- a failing store never fails a prediction
class FailingStore(SpyStore):
    """Raises on the operations named, as a locked SQLite file or an unreachable Redis would."""

    def __init__(self, *fail):
        super().__init__()
        self.fail = set(fail)

    def get(self, key, now):
        if "get" in self.fail:
            raise OSError("store unreachable")
        return super().get(key, now)

    def add(self, items, now):
        if "add" in self.fail:
            raise OSError("database is locked")
        return super().add(items, now)

    def renew(self, key, expires_at, now):
        if "renew" in self.fail:
            raise OSError("database is locked")
        return super().renew(key, expires_at, now)


def quietly(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = call()
    return out, [str(w.message) for w in caught if "DecisionCache: could not" in str(w.message)]


for op in ("get", "add"):
    cache = DecisionCache(store=FailingStore(op))
    r, en, _ = router_with(cache)
    outs, warned = quietly(lambda: [r.predict(STATE, Q) for _ in range(3)])
    check("fail-open/%s raises: every request is still answered" % op,
          [o["answers"]["billing"]["noul"] for o in outs], [0.7] * 3)
    check("fail-open/%s raises: each request runs the model" % op, len(en.calls), 3)
    check("fail-open/%s raises: each failure is counted" % op, cache.cache_info()["errors"], 3)
    check("fail-open/%s raises: it warns once" % op, len(warned), 1)
    check("fail-open/%s raises: no claim outlives its request" % op, cache._inflight, {})

cache = DecisionCache(store=FailingStore("renew"), ttl=160, renew_on_hit=True)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
clock.t += 20  # past a 16th of the ttl, so the hit tries to renew
out, _ = quietly(lambda: r.predict(STATE, Q))
check("fail-open/renew raises: the hit is still replayed", (len(en.calls), out["answers"]["billing"]["noul"]),
      (1, 0.7))
check("fail-open/renew raises: counted", cache.cache_info()["errors"], 1)

spy = SpyStore()
cache = DecisionCache(store=spy)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
for key, (_, expires_at) in list(spy.data.items()):
    spy.data[key] = (b"z not zlib, not JSON", expires_at)
out, warned = quietly(lambda: r.predict(STATE, Q))
check("fail-open/an undecodable decision is computed again", (len(en.calls), out["answers"]["billing"]["noul"]),
      (2, 0.7))
check("fail-open/an undecodable decision is counted and warned about", (cache.cache_info()["errors"], len(warned)),
      (1, 1))
check("fail-open/the new decision replaces the undecodable one", r.predict(STATE, Q)["answers"]["billing"]["noul"],
      0.7)
check("fail-open/and is replayed from then on", len(en.calls), 2)

# --------------------------------------------------------------- SQLite layout
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "forever.sqlite")
    cache = DecisionCache(path)
    r, en, _ = router_with(cache)
    r.predict(STATE, Q)
    cache.close()
    clock.t += 10 ** 7
    cache = DecisionCache(path)
    r, en, _ = router_with(cache)
    r.predict(STATE, Q)
    check("sqlite/a decision without expiry survives a reopen", len(en.calls), 0)
    cache.close()

    import sqlite3
    old = os.path.join(tmp, "old.sqlite")
    db = sqlite3.connect(old)
    db.execute("CREATE TABLE laya_decisions (key BLOB PRIMARY KEY, stored_at REAL NOT NULL, value BLOB NOT NULL)")
    db.close()
    check_raises("sqlite/a file in an older layout is refused, not misread", ValueError, lambda: DecisionCache(old))

# --------------------------------------------------------------- memory store sweeps mixed lifetimes
store = consistency._MemoryStore(maxsize=None)
now = 1000.0
store.add([(b"%d" % i, b"v", now + (5 if i % 2 else 10 ** 6)) for i in range(10)], now)
check("memory/prune drops only the expired among mixed lifetimes", store.prune(now + 6), 5)
check("memory/the rest stay", len(store), 5)


# --------------------------------------------------------------- exports
for name in ("DecisionCache", "DecisionStore", "decision_margins"):
    check_true("export/%s is in __all__" % name, name in laya.__all__)

# --------------------------------------------------------------- the archived Feishu runs
# research/benchmarks/feishu_zh keeps three repeats of 64 cases for Jev and Laya. An answer that
# changed side between Jev's runs must sit inside a 0.1 margin in every run, while that band
# flags only a few percent of all answers; Laya's repeats must be identical.
ARCHIVE = os.path.join(ROOT, "research", "benchmarks", "feishu_zh", "results", "v1")
THRESHOLDS = {"related": 0.5, "action": 0.5, "urgent": 0.75, "value": 0.5}
for backend in ("jev", "laya"):
    with open(os.path.join(ARCHIVE, backend, "raw.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["id"]), []).append(row)
    flagged = unstable = covered = 0
    identical = 0
    for reps in groups.values():
        margins = [decision_margins(rep["response"], THRESHOLDS) for rep in reps]
        flagged += sum(m < 0.1 for per in margins for m in per.values())
        identical += all(rep["response"]["answers"] == reps[0]["response"]["answers"] for rep in reps)
        answers = [rep["response"]["answers"] for rep in reps]
        for name in answers[0]:
            if "noul" in answers[0][name]:
                sides = {a[name]["noul"] >= THRESHOLDS[name] for a in answers}
            else:
                sides = {a[name]["choice"] for a in answers}
            if len(sides) > 1:
                unstable += len(reps)
                covered += sum(per[name] < 0.1 for per in margins)
    total = sum(len(rep["response"]["answers"]) for reps in groups.values() for rep in reps)
    if backend == "jev":
        # 3 answers changed side between runs; counted once per run, that is 9 readings.
        check("archive/jev readings of answers that changed between runs", unstable, 9)
        check("archive/a 0.1 margin flags each of them in every run", covered, unstable)
        check_true("archive/while flagging under 5%% of answers (%d/%d)" % (flagged, total),
                   flagged < 0.05 * total)
        # 38 of the 64 choice answers repeat their probabilities, but 9 of those still change
        # `confidence`, which Jev computes below the precision it prints probabilities at.
        check("archive/jev responses identical across three runs", identical, 29)
    else:
        check("archive/laya never changed an answer", unstable, 0)
        check("archive/laya responses identical across three runs", identical, len(groups))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all consistency tests passed")
sys.exit(1 if FAIL else 0)
