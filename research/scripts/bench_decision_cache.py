"""Cost of every step of `DecisionCache`, with the alternatives it was chosen over.

No model is needed: the Router runs over a fake agent that answers instantly, so what is timed is
the cache and the engine code around it, on real result payloads from the archived Feishu run.

    python research/scripts/bench_decision_cache.py --out research/results/decision_cache_bench.json

Sections: key (canonicalisation + hash, per state size), codec (value encode/decode, sizes),
store (memory vs SQLite get/put, SQLite durability and batching, bytes per entry), write-behind
(an asynchronous SQLite writer vs the synchronous one, on the miss path), and request (the whole
`Router.predict` call with no cache, a miss and a hit).
"""
import argparse
import hashlib
import json
import os
import platform
import queue
import sqlite3
import statistics
import sys
import tempfile
import threading
import timeit
import tracemalloc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import laya  # noqa: E402
from laya import DecisionCache, Router  # noqa: E402
from laya import consistency  # noqa: E402
from laya.hooks import PredictContext  # noqa: E402

ARCHIVE = os.path.join(ROOT, "research", "benchmarks", "feishu_zh", "results", "v1", "laya", "raw.jsonl")


def per_call_us(fn, number, repeat=7):
    """Median and best per-call time in microseconds, over `repeat` runs of `number` calls."""
    runs = timeit.Timer(fn).repeat(repeat=repeat, number=number)
    per = sorted(r / number * 1e6 for r in runs)
    return {"median_us": round(statistics.median(per), 3), "best_us": round(per[0], 3)}


def payloads():
    rows = [json.loads(line) for line in open(ARCHIVE, encoding="utf-8")]
    choice = next(r["response"] for r in rows if r["mode"] == "choice")
    four = next(r["response"] for r in rows if r["mode"] == "four_noul")
    return {"choice_4_options": choice, "four_noul": four}


BASE = ("Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our "
        "plan. Invoice INV-2291 shows two identical charges of $49 on 3 March. ")
STATES = {"short_140_chars": BASE[:140], "medium_2k_chars": (BASE * 12)[:2000],
          "long_20k_chars": (BASE * 120)[:20000]}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}


class FakeAgent:
    """Instant answers with the identity attributes a real Agent carries, for the fingerprint."""

    model_id = "convaiinnovations/laya"
    revision = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

    def __init__(self, payload):
        self.payload = payload
        self.cfg = {"max_len": 512, "head_max_len": 192}
        self.temperature = [1.21, 1.35, 1.08]
        self.temperature_by_options = {"%s:%s" % (t, k): 1.1 for t in ("choice", "score", "noul")
                                       for k in ("2", "3-5", "6-10", "11+")}
        self.lang_temperatures = {}

    def system_one(self, state, questions, **kw):
        return json.loads(json.dumps(self.payload))

    def predict_batch(self, states, questions, batch_size=None, **kw):
        return [self.system_one(s, questions) for s in states]


def router_ctx(state, agent):
    decision = {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
                "detection": {"script": "latin", "language": "en"}, "workflow": None}
    return PredictContext(states=[state], questions=QUESTIONS, decision=decision, model="english",
                          agent=agent, router=object())


# --------------------------------------------------------------------------- key
def bench_key(agent):
    out = {}
    cache = DecisionCache(fingerprint=None)
    for label, state in STATES.items():
        ctx = router_ctx(state, agent)
        prefix = consistency._canonical([1, consistency._fingerprint(agent), "english", None, None, None,
                                         QUESTIONS])
        body = consistency._canonical(state)
        out[label] = {
            "canonical_bytes": len(prefix) + 1 + len(body),
            "DecisionCache._keys (total)": per_call_us(lambda: cache._keys(ctx, ctx.states, False), 2000),
            "fingerprint": per_call_us(lambda: consistency._fingerprint(agent), 20000),
            "canonical json, prefix": per_call_us(lambda: consistency._canonical(
                [1, consistency._fingerprint(agent), "english", None, None, None, QUESTIONS]), 5000),
            "canonical json, state": per_call_us(lambda: consistency._canonical(state), 5000),
            "blake2b-128 over the bytes": per_call_us(
                lambda: hashlib.blake2b(prefix + b"\x00" + body, digest_size=16).digest(), 20000),
            "sha256 over the bytes": per_call_us(lambda: hashlib.sha256(prefix + b"\x00" + body).digest(), 20000),
            "md5 over the bytes (reference only)": per_call_us(
                lambda: hashlib.md5(prefix + b"\x00" + body).digest(), 20000),
        }
    return out


# --------------------------------------------------------------------------- codec
def bench_codec(samples):
    out = {}
    cache = DecisionCache()
    for label, payload in samples.items():
        blob = cache._encode(payload)
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        out[label] = {
            "raw_json_bytes": len(raw),
            "stored_bytes": len(blob),
            "stored_as": "zlib" if blob[:1] == b"z" else "json",
            "encode (json + zlib when smaller)": per_call_us(lambda: cache._encode(payload), 5000),
            "decode": per_call_us(lambda: consistency._unpack(blob), 5000),
            "encode, json only": per_call_us(
                lambda: b"j" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), 5000),
            "decode, json only": per_call_us(lambda: json.loads(raw), 5000),
        }
    return out


# --------------------------------------------------------------------------- stores
def keys_for(n, salt=b""):
    return [hashlib.blake2b(salt + i.to_bytes(8, "little"), digest_size=16).digest() for i in range(n)]


def bench_stores(blob, tmp, n=100_000):
    out = {}
    keys = keys_for(n)
    missing = keys_for(1000, b"missing")

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    mem = consistency._MemoryStore(maxsize=None)
    for key in keys:
        mem.put_many([(key, blob)], 1.0, None)
    mem_bytes = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    it = iter(range(10 ** 9))
    out["memory"] = {
        "entries": n,
        "bytes_per_entry_python_heap": round(mem_bytes / n, 1),
        "get hit": per_call_us(lambda: mem.get(keys[next(it) % n], None), 20000),
        "get miss": per_call_us(lambda: mem.get(missing[next(it) % 1000], None), 20000),
        "put one": per_call_us(lambda: mem.put_many([(missing[next(it) % 1000], blob)], 2.0, None), 20000),
    }

    for sync in ("NORMAL", "FULL"):
        path = os.path.join(tmp, "bench_%s.sqlite" % sync)
        store = consistency._SQLiteStore(path, maxsize=None)
        store._db.execute("PRAGMA synchronous=%s" % sync)
        for start in range(0, n, 1000):
            store.put_many([(k, blob) for k in keys[start:start + 1000]], 1.0, None)
        store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = os.path.getsize(path)
        fresh = iter(keys_for(200_000, b"fresh-%s" % sync.encode()))
        out["sqlite synchronous=%s" % sync] = {
            "entries": n,
            "file_bytes_per_entry": round(size / n, 1),
            "get hit": per_call_us(lambda: store.get(keys[next(it) % n], None), 5000),
            "get miss": per_call_us(lambda: store.get(missing[next(it) % 1000], None), 5000),
            "put one (own transaction)": per_call_us(lambda: store.put_many([(next(fresh), blob)], 2.0, None),
                                                     200, repeat=5),
            "put 100 in one transaction, per decision": {
                k: round(v / 100, 3) for k, v in per_call_us(
                    lambda: store.put_many([(next(fresh), blob) for _ in range(100)], 2.0, None),
                    20, repeat=5).items()},
        }
        store.close()
    return out


# --------------------------------------------------------------------------- write-behind
class WriteBehindStore:
    """What "store asynchronously, do not wait" would look like: a queue and a writer thread.

    Reads check the not-yet-written decisions first, so this process sees its own writes at once.
    Another process only sees them once the writer commits, so two processes can each return
    their own decision for the same request in that window: first-decision-wins becomes eventual.
    """

    def __init__(self, path):
        self.inner = consistency._SQLiteStore(path, maxsize=None)
        self.pending = {}
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            batch = [self.q.get()]
            while True:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            if None in batch:
                return
            with self.lock:
                items = [(k, self.pending[k]) for k in batch if k in self.pending]
            self.inner.put_many(items, 3.0, None)
            with self.lock:
                for k, _ in items:
                    self.pending.pop(k, None)

    def get(self, key, cutoff):
        with self.lock:
            blob = self.pending.get(key)
        return blob if blob is not None else self.inner.get(key, cutoff)

    def put_many(self, items, now, cutoff):
        with self.lock:
            for key, blob in items:
                self.pending.setdefault(key, blob)
        for key, _ in items:
            self.q.put(key)
        return [blob for _, blob in items]

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.inner.close()


def bench_write_behind(blob, tmp):
    out = {}
    sync_store = consistency._SQLiteStore(os.path.join(tmp, "sync.sqlite"), maxsize=None)
    async_store = WriteBehindStore(os.path.join(tmp, "async.sqlite"))
    for label, store in (("synchronous (shipped)", sync_store), ("write-behind thread", async_store)):
        fresh = iter(keys_for(100_000, label.encode()))
        out[label] = {"store one decision on the request path": per_call_us(
            lambda: store.put_many([(next(fresh), blob)], 2.0, None), 500, repeat=5)}
    async_store.close()
    sync_store.close()
    return out


# --------------------------------------------------------------------------- request
def bench_request(payload, tmp):
    out = {}
    state = STATES["short_140_chars"]
    plain = Router()
    plain.attach("english", FakeAgent(payload))
    out["Router.predict, no cache"] = per_call_us(lambda: plain.predict(state, QUESTIONS), 2000)
    for label, path in (("memory", None), ("sqlite", os.path.join(tmp, "request.sqlite"))):
        cache = DecisionCache(path)
        router = Router(hooks=[cache])
        router.attach("english", FakeAgent(payload))
        router.predict(state, QUESTIONS)
        out["Router.predict, %s cache hit" % label] = per_call_us(lambda: router.predict(state, QUESTIONS), 2000)
        n = iter(range(10 ** 9))
        out["Router.predict, %s cache miss" % label] = per_call_us(
            lambda: router.predict("%s %d" % (state, next(n)), QUESTIONS), 300, repeat=5)
        cache.close()
    for label, s in STATES.items():
        out["routing only: Router.route, %s" % label] = per_call_us(lambda: plain.route(s, QUESTIONS), 500)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="write the results here as JSON")
    parser.add_argument("--entries", type=int, default=100_000, help="decisions held by the stores")
    args = parser.parse_args()
    samples = payloads()
    agent = FakeAgent(samples["four_noul"])
    blob = DecisionCache()._encode(samples["four_noul"])
    with tempfile.TemporaryDirectory() as tmp:
        results = {
            "meta": {"python": platform.python_version(), "machine": platform.machine(),
                     "processor": platform.processor() or platform.platform(), "cpus": os.cpu_count(),
                     "sqlite": sqlite3.sqlite_version, "laya": laya.__version__},
            "key": bench_key(agent),
            "codec": bench_codec(samples),
            "store": bench_stores(blob, tmp, n=args.entries),
            "write_behind": bench_write_behind(blob, tmp),
            "request": bench_request(samples["four_noul"], tmp),
        }
    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
