"""Decision consistency: how close an answer is to flipping, and a cache that replays decisions.

A Laya forward pass samples nothing, so the same request on the same checkpoint and hardware gets
the same answer, byte for byte. A decision can still move with what surrounds the pass: other
hardware or precision, a new checkpoint revision or calibration, or a request that differs in a
detail. Two tools cover that:

* `decision_margins` reads a result and says how far each answer sits from its decision boundary.
  It needs only the payload, so it reads the same schema from a Jev-compatible API as well.
* `DecisionCache` is a prediction hook that stores each decision under a hash of the request and
  the model that answered it, and replays it when the same request comes again.

Both are plain Python: importing this module must not pull in torch.
"""
from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
import warnings
import weakref
import zlib
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .hooks import PredictContext, aggregate_usage

# Part of every key: bump it when the key layout changes, so no entry stored under an older
# layout is ever replayed.
_KEY_VERSION = 1
# A stored value at least this long is zlib-compressed when that makes it smaller.
_COMPRESS_MIN = 256
# Wall clock rather than a monotonic one, because a persisted decision has to age across restarts.
_now = time.time
_STAMP = struct.Struct("<d")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN has no distance to a boundary


def _threshold(value: Any) -> float:
    number = _number(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise ValueError("noul_threshold must be a probability in [0, 1], got %r" % (value,))
    return number


def decision_margins(result: Mapping[str, Any],
                     noul_threshold: Union[float, Mapping[str, float]] = 0.5) -> Dict[str, float]:
    """How far each answer is from flipping, from 0 (on the boundary) to 1 (as far as it gets).

    For `choice` and `score` answers the margin is the probability gap between the two most likely
    options. For `noul` answers it is twice the distance from P(yes) to the threshold the caller
    acts on, capped at 1; at the default threshold of 0.5 that is the gap between P(yes) and P(no),
    so every type is on the same scale.

    A small margin means a small change can flip the answer: another run of a model whose output
    varies between runs, other hardware or precision, a reworded request or a recalibration. The
    margin is read from the payload alone, so it works on Laya's answers, on answers a
    `DecisionCache` replays, and on the same schema from a Jev-compatible API:

        margins = decision_margins(router.predict(state, questions))
        borderline = [name for name, margin in margins.items() if margin < 0.1]

    Args:
        result: A prediction result, or its `answers` mapping.
        noul_threshold: The P(yes) a caller acts on: one value for every `noul` answer, or a
            mapping from question name to threshold in which a missing name means 0.5.

    Returns:
        `{question name: margin}`, rounded to 4 decimals, for every answer that carries
        probabilities or a `noul` value. Other entries are left out.
    """
    answers = result.get("answers", result) if isinstance(result, Mapping) else None
    if not isinstance(answers, Mapping):
        raise TypeError("decision_margins expects a result or its answers mapping, got %s"
                        % type(result).__name__)
    default = None if isinstance(noul_threshold, Mapping) else _threshold(noul_threshold)
    margins: Dict[str, float] = {}
    for name, answer in answers.items():
        if not isinstance(answer, Mapping):
            continue
        if "noul" in answer:
            p = _number(answer["noul"])
            if p is None:
                continue
            t = default if default is not None else _threshold(noul_threshold.get(name, 0.5))
            margins[name] = round(min(1.0, 2.0 * abs(p - t)), 4)
            continue
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, Mapping):
            continue
        top = sorted((p for p in map(_number, probabilities.values()) if p is not None), reverse=True)
        if top:
            margins[name] = round(top[0] - top[1], 4) if len(top) > 1 else 1.0
    return margins


def _canonical(value: Any) -> bytes:
    # Order-preserving at every level, as in the Router's `_question_schema`: option order is
    # positional, so two schemas that differ only in order are different requests (#166).
    # ASCII output keeps any string, lone surrogates included, encodable.
    return json.dumps(value, sort_keys=False, separators=(",", ":"), default=str).encode("ascii")


def _pack(raw: bytes) -> bytes:
    if len(raw) >= _COMPRESS_MIN:
        packed = zlib.compress(raw, 6)
        if len(packed) < len(raw):
            return b"z" + packed
    return b"j" + raw


def _unpack(blob: bytes) -> Dict[str, Any]:
    raw = zlib.decompress(blob[1:]) if blob[:1] == b"z" else blob[1:]
    return json.loads(raw.decode("utf-8", "surrogatepass"))


def _fingerprint(agent: Any) -> List[Any]:
    """What identifies the answering model: when any of it changes, old decisions must not replay."""
    from . import __version__

    cfg = getattr(agent, "cfg", None)
    if not isinstance(cfg, Mapping):
        cfg = {}
    return [__version__, getattr(agent, "model_id", None), getattr(agent, "revision", None),
            cfg.get("max_len"), cfg.get("head_max_len"),
            getattr(agent, "temperature", None), getattr(agent, "temperature_by_options", None),
            getattr(agent, "lang_temperatures", None)]


class _MemoryStore:
    """Decisions in this process: key -> 8-byte stored-at stamp + value, oldest first."""

    def __init__(self, maxsize: Optional[int]):
        self.maxsize = maxsize
        self._data: "OrderedDict[bytes, bytes]" = OrderedDict()

    def get(self, key: bytes, cutoff: Optional[float]) -> Optional[bytes]:
        entry = self._data.get(key)
        if entry is None:
            return None
        if cutoff is not None and _STAMP.unpack_from(entry)[0] <= cutoff:
            del self._data[key]
            return None
        return entry[_STAMP.size:]

    def put_many(self, items: Sequence[Tuple[bytes, bytes]], now: float,
                 cutoff: Optional[float]) -> List[bytes]:
        stamp = _STAMP.pack(now)
        stored = []
        for key, blob in items:
            first = self.get(key, cutoff)
            if first is None:
                # get() dropped an expired entry, so this appends: the dict stays oldest first.
                self._data[key] = stamp + blob
                first = blob
            stored.append(first)
        # Oldest first means the expired entries are at the front, so trimming them is cheap.
        data = self._data
        while data and cutoff is not None and _STAMP.unpack_from(next(iter(data.values())))[0] <= cutoff:
            data.popitem(last=False)
        while self.maxsize is not None and len(data) > self.maxsize:
            data.popitem(last=False)
        return stored

    def prune(self, cutoff: Optional[float]) -> int:
        expired = [] if cutoff is None else [
            key for key, entry in self._data.items() if _STAMP.unpack_from(entry)[0] <= cutoff]
        for key in expired:
            del self._data[key]
        dropped = len(expired)
        while self.maxsize is not None and len(self._data) > self.maxsize:
            self._data.popitem(last=False)
            dropped += 1
        return dropped

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def close(self) -> None:
        pass


class _SQLiteStore:
    """Decisions in a SQLite file, shared by every process that opens it."""

    def __init__(self, path: Any, maxsize: Optional[int]):
        import sqlite3  # here rather than at import time: some Python builds ship without it

        self.maxsize = maxsize
        # COUNT(*) walks the table, so the size bound is checked every so many stores.
        self._every = 1000 if maxsize is None else max(1, min(1000, maxsize // 10))
        self._stores = 0
        self._db = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS laya_decisions (key BLOB PRIMARY KEY, "
                         "stored_at REAL NOT NULL, value BLOB NOT NULL) WITHOUT ROWID")
        self._db.execute("CREATE INDEX IF NOT EXISTS laya_decisions_stored_at ON laya_decisions (stored_at)")

    def get(self, key: bytes, cutoff: Optional[float]) -> Optional[bytes]:
        row = self._db.execute("SELECT stored_at, value FROM laya_decisions WHERE key = ?", (key,)).fetchone()
        if row is None or (cutoff is not None and row[0] <= cutoff):
            return None
        return bytes(row[1])

    def put_many(self, items: Sequence[Tuple[bytes, bytes]], now: float,
                 cutoff: Optional[float]) -> List[bytes]:
        stored = []
        # IMMEDIATE takes the write lock up front, so two processes cannot both see a key as
        # missing and both store it: the second one reads the first one's decision.
        self._db.execute("BEGIN IMMEDIATE")
        try:
            for key, blob in items:
                first = self.get(key, cutoff)
                if first is None:
                    self._db.execute("INSERT OR REPLACE INTO laya_decisions (key, stored_at, value) "
                                     "VALUES (?, ?, ?)", (key, now, blob))
                    self._stores += 1
                    first = blob
                stored.append(first)
            if self._stores >= self._every and (cutoff is not None or self.maxsize is not None):
                self._stores = 0
                self._prune(cutoff)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return stored

    def _prune(self, cutoff: Optional[float]) -> int:
        dropped = 0
        if cutoff is not None:
            dropped += self._db.execute("DELETE FROM laya_decisions WHERE stored_at <= ?", (cutoff,)).rowcount
        if self.maxsize is not None:
            (count,) = self._db.execute("SELECT COUNT(*) FROM laya_decisions").fetchone()
            if count > self.maxsize:
                dropped += self._db.execute(
                    "DELETE FROM laya_decisions WHERE key IN "
                    "(SELECT key FROM laya_decisions ORDER BY stored_at LIMIT ?)",
                    (count - self.maxsize,)).rowcount
        return dropped

    def prune(self, cutoff: Optional[float]) -> int:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            dropped = self._prune(cutoff)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return dropped

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM laya_decisions").fetchone()[0]

    def clear(self) -> None:
        self._db.execute("DELETE FROM laya_decisions")

    def close(self) -> None:
        self._db.close()


class DecisionCache:
    """A prediction hook that replays the decision a request got the first time.

    Install it on a `Router`, where it keys on the routed checkpoint and on the language the
    request is scored in, or on an `Agent` or `ONNXAgent`:

        from laya import DecisionCache, Router

        cache = DecisionCache("decisions.sqlite", ttl=30 * 24 * 3600)
        router = Router(hooks=[cache])

    Before inference it looks every state up. A state that hits skips the forward pass; the rest
    run as usual. After inference it stores what the model returned. The first decision stored
    for a request wins: a request that missed at the same time, in another thread or in another
    process sharing the file, returns the stored decision instead of its own, so every caller
    gets one answer per request while the entry lives. A replay is the stored payload exactly,
    `usage` included; a `Router` adds the current `routing` to it.

    The key is a 16-byte BLAKE2b hash of the state, the questions in their given order (option
    order is positional), the per-call token budget, the checkpoint and the model's fingerprint,
    plus the routing decision when the checkpoint has per-language temperatures, the one way
    the language reaches an answer. Values are compact JSON, compressed when that is smaller;
    nothing of the request text is stored. Decisions leave oldest first: once they are `ttl`
    seconds old, and when more than `maxsize` are held. A hit writes nothing; only a miss stores.

    An `Agent` call may pass `lang`, which a hook cannot see. On an `Agent` with per-language
    temperatures the cache therefore stands aside rather than risk replaying an answer scored for
    another language; install it on the `Router` there.

    Args:
        path: SQLite file that keeps decisions across restarts and shares them between processes.
            `None` keeps them in this process's memory.
        ttl: Seconds a decision is replayed; after that the next identical request is answered
            afresh. `None` never expires a decision.
        maxsize: Most decisions held, the oldest leaving first; `None` for no bound. With a
            `path`, the bound is enforced every `min(1000, maxsize // 10)` stores.
        fingerprint: The model's part of the key. `None` derives it from the checkpoint id,
            revision, calibration temperatures, token budget and Laya version, so a change to any
            of them starts a fresh set of decisions. A fixed string keeps replaying decisions
            across such changes until they expire.
    """

    def __init__(self, path: Optional[str] = None, *, ttl: Optional[float] = None,
                 maxsize: Optional[int] = 100_000, fingerprint: Optional[str] = None):
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not ttl > 0):
            raise ValueError("ttl must be a positive number of seconds or None, got %r" % (ttl,))
        if maxsize is not None and (isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1):
            raise ValueError("maxsize must be a positive integer or None, got %r" % (maxsize,))
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise TypeError("fingerprint must be a string or None, got %s" % type(fingerprint).__name__)
        self.path = path
        self.ttl = None if ttl is None else float(ttl)
        self.maxsize = maxsize
        self.fingerprint = fingerprint
        self._store = _MemoryStore(maxsize) if path is None else _SQLiteStore(path, maxsize)
        self._lock = threading.Lock()
        self._pending: "weakref.WeakKeyDictionary[PredictContext, tuple]" = weakref.WeakKeyDictionary()
        self._counts = {"hits": 0, "misses": 0, "conflicts": 0}
        self._warned = False

    def __repr__(self) -> str:
        return "DecisionCache(path=%r, ttl=%r, maxsize=%r)" % (self.path, self.ttl, self.maxsize)

    def _cutoff(self, now: float) -> Optional[float]:
        return None if self.ttl is None else now - self.ttl

    def _keys(self, ctx: PredictContext, states: Sequence[Any], lang_sensitive: bool) -> List[bytes]:
        model = self.fingerprint if self.fingerprint is not None else _fingerprint(ctx.agent)
        # The language only reaches the answer through per-language temperatures. When the agent
        # has them, key on the whole routing decision: its reason records an explicit `lang` and
        # its detection the detected one. Otherwise requests routed to the same checkpoint by
        # different means get the same answer, so they share an entry.
        route = ctx.decision if lang_sensitive else None
        base = hashlib.blake2b(_canonical([_KEY_VERSION, model, ctx.model, route, ctx.max_len,
                                           ctx.head_max_len, ctx.questions]), digest_size=16)
        keys = []
        for state in states:
            key = base.copy()
            key.update(b"\x00")  # a byte canonical JSON never contains, between prefix and state
            key.update(_canonical(state))
            keys.append(key.digest())
        return keys

    def _encode(self, result: Any) -> Optional[bytes]:
        if not isinstance(result, dict):
            return None
        # The Router re-adds the current `routing` to a replay, so it is not stored.
        payload = {k: v for k, v in result.items() if k != "routing"}
        try:
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            if not self._warned:
                self._warned = True
                warnings.warn("laya: DecisionCache: a result is not plain JSON, so it cannot be replayed "
                              "exactly and is not cached (%s)" % exc, RuntimeWarning, stacklevel=3)
            return None
        return _pack(raw.encode("utf-8", "surrogatepass"))

    def on_predict_start(self, ctx: PredictContext) -> None:
        states = ctx.states
        if ctx.results is not None or not ctx.questions or not isinstance(states, (list, tuple)) or not states:
            return
        lang_sensitive = bool(getattr(ctx.agent, "lang_temperatures", None))
        if lang_sensitive and ctx.router is None:
            return
        keys = self._keys(ctx, states, lang_sensitive)
        with self._lock:
            cutoff = self._cutoff(_now())
            blobs = [self._store.get(key, cutoff) for key in keys]
            hits = sum(blob is not None for blob in blobs)
            self._counts["hits"] += hits
            self._counts["misses"] += len(keys) - hits
        replays = [None if blob is None else _unpack(blob) for blob in blobs]
        if hits == len(keys):
            ctx.skip(replays)
            return
        misses = [i for i, replay in enumerate(replays) if replay is None]
        with self._lock:
            self._pending[ctx] = (states, keys, replays, misses)
        if hits:
            # Only the states that missed reach the model; the end hook puts the rest back.
            ctx.states = [states[i] for i in misses]

    def on_predict_end(self, ctx: PredictContext) -> None:
        with self._lock:
            pending = self._pending.pop(ctx, None)
        if pending is None:
            return
        states, keys, replays, misses = pending
        partial = len(misses) < len(keys)
        if partial:
            ctx.states = states
        results = ctx.results
        if ctx.error is not None or not isinstance(results, list) or len(results) != len(misses):
            return
        blobs = [self._encode(result) for result in results]
        items = [(keys[i], blob) for i, blob in zip(misses, blobs) if blob is not None]
        with self._lock:
            now = _now()
            stored = iter(self._store.put_many(items, now, self._cutoff(now)) if items else ())
        merged = list(replays)
        conflicts = 0
        for i, result, blob in zip(misses, results, blobs):
            if blob is not None:
                first = next(stored)
                if first != blob:
                    # Another run stored this request first, possibly on other hardware: replay
                    # its decision, so every caller of this request gets the same one.
                    replay = _unpack(first)
                    if isinstance(result, dict) and "routing" in result:
                        replay["routing"] = result["routing"]
                    result = replay
                    conflicts += 1
            merged[i] = result
        if conflicts:
            with self._lock:
                self._counts["conflicts"] += conflicts
        if partial or conflicts:
            ctx.results = merged
            ctx.usage = aggregate_usage(merged)

    def prune(self) -> int:
        """Drop every expired decision now, and the oldest beyond `maxsize`.

        Returns:
            How many decisions were dropped.
        """
        with self._lock:
            return self._store.prune(self._cutoff(_now()))

    def cache_info(self) -> Dict[str, Any]:
        """Counters since construction or the last `cache_clear()`.

        Returns:
            A dict with `size` (decisions held), `maxsize`, `hits` and `misses` (per state
            looked up), and `conflicts`: decisions computed for a request that another run had
            already stored differently, which were replaced by the stored one.
        """
        with self._lock:
            return {"size": len(self._store), "maxsize": self.maxsize, **self._counts}

    def cache_clear(self) -> None:
        """Forget every stored decision and reset the counters."""
        with self._lock:
            self._store.clear()
            for name in self._counts:
                self._counts[name] = 0

    def close(self) -> None:
        """Close the SQLite file, if there is one. Do not use the cache afterwards."""
        with self._lock:
            self._store.close()
