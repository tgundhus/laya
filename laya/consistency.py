"""Decision consistency: how close an answer is to flipping, and a cache that replays decisions.

A Laya forward pass samples nothing, so the same request on the same checkpoint and hardware gets
the same answer, byte for byte. A decision can still move with what surrounds the pass: other
hardware or precision, a new checkpoint revision or calibration, or a request that differs in a
detail. Two tools cover that:

* `decision_margins` reads a result and says how far each answer sits from its decision boundary.
  It needs only the payload, so it reads the same schema from a Jev-compatible API as well.
* `DecisionCache` is a prediction hook that stores each decision under a hash of the request and
  the model that answered it, and replays it when the same request comes again. It keeps them in
  memory or in a SQLite file, or in any `DecisionStore`, which is how several machines can share
  one set of decisions.

All of it is plain Python: importing this module must not pull in torch.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
import time
import warnings
import weakref
import zlib
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

from .hooks import PredictContext, aggregate_usage

# Part of every key: bump it when the key layout changes, so no entry stored under an older
# layout is ever replayed.
_KEY_VERSION = 1
# A stored value at least this long is zlib-compressed when that makes it smaller.
_COMPRESS_MIN = 256
# Wall clock rather than a monotonic one, because a persisted decision has to age across restarts.
_now = time.time
# Held decisions are bounded to this many unless a `maxsize` says otherwise.
DEFAULT_MAXSIZE = 100_000
# A renewal is written only once a decision has aged this share of its lifetime, so a request that
# keeps coming costs at most 16 renewal writes per lifetime rather than one per hit.
_RENEW_FRACTION = 1 / 16


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


class DecisionStore(Protocol):
    """Where a `DecisionCache` keeps its decisions: bytes values under 16-byte keys, each with an expiry.

    The cache does the hashing, encoding and retention policy; a store only keeps bytes. Implement
    one to share decisions between machines -- over Redis or Postgres, say -- and pass it as
    `DecisionCache(store=...)`. Times are Unix seconds, and `math.inf` means never. A store is
    called from several threads at once and must be safe for that.

    The rule that matters is in `add`: when a key already holds a value that has not expired,
    the store keeps that value and returns it. That is what makes the first decision stored for a
    request the one every caller gets, across threads, processes and machines.
    """

    def get(self, key: bytes, now: float) -> Optional[Tuple[bytes, float]]:
        """The value under `key` and when it expires, or None when it holds none unexpired at `now`."""
        ...

    def add(self, items: Sequence[Tuple[bytes, bytes, float]], now: float) -> List[bytes]:
        """Store each `(key, value, expires_at)` unless `key` already holds an unexpired value.

        Returns:
            For each item, in order, the value its key holds afterwards: the one just stored, or
            the one that was there first.
        """
        ...

    def renew(self, key: bytes, expires_at: float, now: float) -> None:
        """Move the expiry of the value under `key`, if it holds one, to `expires_at`."""
        ...

    def prune(self, now: float) -> int:
        """Drop what has expired at `now`, and anything over the store's own bound; return how many."""
        ...

    def __len__(self) -> int:
        """How many values are held, counting expired ones not yet dropped."""
        ...

    def clear(self) -> None:
        """Drop every value."""
        ...

    def close(self) -> None:
        """Release what the store holds open."""
        ...


_STORE_METHODS = ("get", "add", "renew", "prune", "__len__", "clear", "close")
_ENTRY = struct.Struct("<dd")  # touched at, expires at


class _MemoryStore:
    """Decisions in this process, least recently stored or renewed first."""

    def __init__(self, maxsize: Optional[int]):
        self.maxsize = maxsize
        self._data: "OrderedDict[bytes, bytes]" = OrderedDict()
        self._lock = threading.Lock()
        self._adds_until_sweep = 4096

    def _get(self, key: bytes, now: float) -> Optional[Tuple[bytes, float]]:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at = _ENTRY.unpack_from(entry)[1]
        if expires_at <= now:
            del self._data[key]
            return None
        return entry[_ENTRY.size:], expires_at

    def _sweep(self, now: float) -> int:
        expired = [key for key, entry in self._data.items() if _ENTRY.unpack_from(entry)[1] <= now]
        for key in expired:
            del self._data[key]
        # Lifetimes differ per decision, so expired ones are anywhere in the order: sweep them all,
        # but only after as many stores as would make the sweep cheap on average.
        self._adds_until_sweep = max(4096, len(self._data) // 4)
        return len(expired)

    def get(self, key: bytes, now: float) -> Optional[Tuple[bytes, float]]:
        with self._lock:
            return self._get(key, now)

    def add(self, items: Sequence[Tuple[bytes, bytes, float]], now: float) -> List[bytes]:
        held = []
        with self._lock:
            for key, value, expires_at in items:
                first = self._get(key, now)
                if first is None:
                    self._data[key] = _ENTRY.pack(now, expires_at) + value
                    self._adds_until_sweep -= 1
                    first = (value, expires_at)
                held.append(first[0])
            while self.maxsize is not None and len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            if self._adds_until_sweep <= 0:
                self._sweep(now)
        return held

    def renew(self, key: bytes, expires_at: float, now: float) -> None:
        with self._lock:
            entry = self._data.get(key)
            if entry is not None:
                self._data[key] = _ENTRY.pack(now, expires_at) + entry[_ENTRY.size:]
                self._data.move_to_end(key)

    def prune(self, now: float) -> int:
        with self._lock:
            dropped = self._sweep(now)
            while self.maxsize is not None and len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                dropped += 1
            return dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def close(self) -> None:
        pass


class _SQLiteStore:
    """Decisions in a SQLite file, shared by every process that opens it."""

    def __init__(self, path: Any, maxsize: Optional[int]):
        import sqlite3  # here rather than at import time: some Python builds ship without it

        self.maxsize = maxsize
        # COUNT(*) walks the table, so the size bound and expiry are enforced every so many stores.
        self._every = 1000 if maxsize is None else max(1, min(1000, maxsize // 10))
        self._adds = 0
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(laya_decisions)")}
        if columns and "expires_at" not in columns:
            self._db.close()
            raise ValueError("%r holds decisions in an older layout; delete it or pass another path" % (path,))
        self._db.execute("CREATE TABLE IF NOT EXISTS laya_decisions (key BLOB PRIMARY KEY, "
                         "touched_at REAL NOT NULL, expires_at REAL NOT NULL, value BLOB NOT NULL) WITHOUT ROWID")
        self._db.execute("CREATE INDEX IF NOT EXISTS laya_decisions_expires_at ON laya_decisions (expires_at)")
        self._db.execute("CREATE INDEX IF NOT EXISTS laya_decisions_touched_at ON laya_decisions (touched_at)")

    def _get(self, key: bytes, now: float) -> Optional[Tuple[bytes, float]]:
        row = self._db.execute("SELECT value, expires_at FROM laya_decisions WHERE key = ?", (key,)).fetchone()
        if row is None or row[1] <= now:
            return None
        return bytes(row[0]), row[1]

    def _prune(self, now: float) -> int:
        dropped = self._db.execute("DELETE FROM laya_decisions WHERE expires_at <= ?", (now,)).rowcount
        if self.maxsize is not None:
            (count,) = self._db.execute("SELECT COUNT(*) FROM laya_decisions").fetchone()
            if count > self.maxsize:
                dropped += self._db.execute(
                    "DELETE FROM laya_decisions WHERE key IN "
                    "(SELECT key FROM laya_decisions ORDER BY touched_at LIMIT ?)",
                    (count - self.maxsize,)).rowcount
        return dropped

    def _transaction(self, work: Callable[[], Any]) -> Any:
        with self._lock:
            # IMMEDIATE takes the write lock up front, so two processes cannot both see a key as
            # missing and both store it: the second one reads the first one's decision.
            self._db.execute("BEGIN IMMEDIATE")
            try:
                result = work()
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            return result

    def get(self, key: bytes, now: float) -> Optional[Tuple[bytes, float]]:
        with self._lock:
            return self._get(key, now)

    def add(self, items: Sequence[Tuple[bytes, bytes, float]], now: float) -> List[bytes]:
        def work():
            held = []
            for key, value, expires_at in items:
                first = self._get(key, now)
                if first is None:
                    self._db.execute("INSERT OR REPLACE INTO laya_decisions (key, touched_at, expires_at, value) "
                                     "VALUES (?, ?, ?, ?)", (key, now, expires_at, value))
                    self._adds += 1
                    first = (value, expires_at)
                held.append(first[0])
            if self._adds >= self._every:
                self._adds = 0
                self._prune(now)
            return held
        return self._transaction(work)

    def renew(self, key: bytes, expires_at: float, now: float) -> None:
        with self._lock:
            self._db.execute("UPDATE laya_decisions SET touched_at = ?, expires_at = ? WHERE key = ? "
                             "AND expires_at > ?", (now, expires_at, key, now))

    def prune(self, now: float) -> int:
        return self._transaction(lambda: self._prune(now))

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM laya_decisions").fetchone()[0]

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM laya_decisions")

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _check_ttl(ttl: Any, where: str) -> Optional[float]:
    if ttl is None:
        return None
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl != ttl:
        raise TypeError("%s must be a number of seconds or None, got %r" % (where, ttl))
    return float(ttl)


class DecisionCache:
    """A prediction hook that replays the decision a request got the first time.

    Install it on a `Router`, where it keys on the routed checkpoint and on the language the
    request is scored in, or on an `Agent` or `ONNXAgent`:

        from laya import DecisionCache, Router

        cache = DecisionCache("decisions.sqlite", ttl=30 * 24 * 3600, renew_on_hit=True)
        router = Router(hooks=[cache])

    Before inference it looks every state up. A state that hits skips the forward pass; the rest
    run as usual. After inference it stores what the model returned. The first decision stored
    for a request wins: a request that missed at the same time, in another thread, process or
    machine sharing the store, returns the stored decision instead of its own, so every caller
    gets one answer per request while the entry lives. A replay is the stored payload exactly,
    `usage` included; a `Router` adds the current `routing` to it.

    The key is a 16-byte BLAKE2b hash of the state, the questions in their given order (option
    order is positional), the per-call token budget, the checkpoint and the model's fingerprint,
    plus the routing decision when the checkpoint has per-language temperatures, the one way
    the language reaches an answer. Values are compact JSON, compressed when that is smaller;
    nothing of the request text is stored.

    Each decision carries its own expiry. `ttl` sets it: a number of seconds, None for never, or
    a function `ttl(questions, result)` that returns either -- or 0 to keep that decision not at
    all -- so a question set or an answer can have its own retention. With `renew_on_hit`, a
    replay pushes the expiry out again, so a decision lasts as long as its request keeps coming;
    the renewal is written at most 16 times per lifetime. Otherwise a hit writes nothing. The
    built-in stores drop expired decisions and, past `maxsize`, the least recently stored or
    renewed ones.

    An `Agent` call may pass `lang`, which a hook cannot see. On an `Agent` with per-language
    temperatures the cache therefore stands aside rather than risk replaying an answer scored for
    another language; install it on the `Router` there.

    Args:
        path: SQLite file that keeps decisions across restarts and shares them between processes.
            `None` keeps them in this process's memory, unless `store` is given.
        ttl: Seconds a decision is replayed, None for no expiry, or a function of
            `(questions, result)` returning seconds, None, or 0 to not keep that decision. It must
            not change `result`.
        maxsize: Most decisions the built-in stores hold; `None` for no bound. With a `path`, the
            bound is enforced every `min(1000, maxsize // 10)` stores. A custom `store` bounds
            itself.
        fingerprint: The model's part of the key. `None` derives it from the checkpoint id,
            revision, calibration temperatures, token budget and Laya version, so a change to any
            of them starts a fresh set of decisions. A fixed string keeps replaying decisions
            across such changes until they expire.
        renew_on_hit: Count a decision's lifetime from its last replay instead of from when it
            was stored.
        store: A `DecisionStore` to keep decisions in instead of memory or `path`, for instance
            one shared by several machines.
    """

    def __init__(self, path: Optional[str] = None, *, ttl: Union[None, float, Callable[..., Any]] = None,
                 maxsize: Optional[int] = DEFAULT_MAXSIZE, fingerprint: Optional[str] = None,
                 renew_on_hit: bool = False, store: Optional[DecisionStore] = None):
        if ttl is not None and not callable(ttl):
            if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not ttl > 0:
                raise ValueError("ttl must be a positive number of seconds, a function or None, got %r" % (ttl,))
            ttl = float(ttl)
        if maxsize is not None and (isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1):
            raise ValueError("maxsize must be a positive integer or None, got %r" % (maxsize,))
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise TypeError("fingerprint must be a string or None, got %s" % type(fingerprint).__name__)
        if store is not None:
            if path is not None:
                raise ValueError("pass a path or a store, not both")
            if maxsize != DEFAULT_MAXSIZE:
                raise ValueError("maxsize bounds the built-in stores; bound a custom store yourself")
            missing = [name for name in _STORE_METHODS if not callable(getattr(store, name, None))]
            if missing:
                raise TypeError("store is missing %s; see laya.consistency.DecisionStore" % ", ".join(missing))
            self._store = store
            self.maxsize = None
        else:
            self._store = _MemoryStore(maxsize) if path is None else _SQLiteStore(path, maxsize)
            self.maxsize = maxsize
        self.path = path
        self.ttl = ttl
        self.fingerprint = fingerprint
        self.renew_on_hit = bool(renew_on_hit)
        self._lock = threading.Lock()
        self._pending: "weakref.WeakKeyDictionary[PredictContext, tuple]" = weakref.WeakKeyDictionary()
        self._counts = {"hits": 0, "misses": 0, "conflicts": 0}
        self._warned = False

    def __repr__(self) -> str:
        where = self.path if self.path is not None else type(self._store).__name__
        return "DecisionCache(%r, ttl=%r, maxsize=%r)" % (where, self.ttl, self.maxsize)

    def _lifetime(self, questions: Any, result: Dict[str, Any]) -> Optional[float]:
        """Seconds this decision is kept: None for ever, 0 or less for not at all."""
        if callable(self.ttl):
            return _check_ttl(self.ttl(questions, result), "ttl(questions, result)")
        return self.ttl

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

    def _encode(self, payload: Dict[str, Any]) -> Optional[bytes]:
        try:
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            if not self._warned:
                self._warned = True
                warnings.warn("laya: DecisionCache: a result is not plain JSON, so it cannot be replayed "
                              "exactly and is not cached (%s)" % exc, RuntimeWarning, stacklevel=3)
            return None
        return _pack(raw.encode("utf-8", "surrogatepass"))

    def _renew(self, ctx: PredictContext, keys: List[bytes], held: List[Any], replays: List[Any],
               now: float) -> None:
        for key, entry, replay in zip(keys, held, replays):
            if entry is None:
                continue
            lifetime = self._lifetime(ctx.questions, replay)
            if lifetime is None or lifetime <= 0:
                continue
            expires_at = now + lifetime
            if expires_at - entry[1] > lifetime * _RENEW_FRACTION:
                self._store.renew(key, expires_at, now)

    def on_predict_start(self, ctx: PredictContext) -> None:
        states = ctx.states
        if ctx.results is not None or not ctx.questions or not isinstance(states, (list, tuple)) or not states:
            return
        lang_sensitive = bool(getattr(ctx.agent, "lang_temperatures", None))
        if lang_sensitive and ctx.router is None:
            return
        keys = self._keys(ctx, states, lang_sensitive)
        now = _now()
        held = [self._store.get(key, now) for key in keys]
        hits = sum(entry is not None for entry in held)
        with self._lock:
            self._counts["hits"] += hits
            self._counts["misses"] += len(keys) - hits
        replays = [None if entry is None else _unpack(entry[0]) for entry in held]
        if hits and self.renew_on_hit:
            self._renew(ctx, keys, held, replays, now)
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
        now = _now()
        items, positions = [], []
        for n, (i, result) in enumerate(zip(misses, results)):
            if not isinstance(result, dict):
                continue
            # The Router re-adds the current `routing` to a replay, so it is not stored.
            payload = {k: v for k, v in result.items() if k != "routing"}
            lifetime = self._lifetime(ctx.questions, payload)
            if lifetime is not None and lifetime <= 0:
                continue
            value = self._encode(payload)
            if value is None:
                continue
            items.append((keys[i], value, math.inf if lifetime is None else now + lifetime))
            positions.append(n)
        held = self._store.add(items, now) if items else []
        merged = list(replays)
        for i, result in zip(misses, results):
            merged[i] = result
        conflicts = 0
        for n, (_, value, _), first in zip(positions, items, held):
            if first != value:
                # Another run stored this request first, possibly on other hardware: replay its
                # decision, so every caller of this request gets the same one.
                i = misses[n]
                replay = _unpack(first)
                if isinstance(results[n], dict) and "routing" in results[n]:
                    replay["routing"] = results[n]["routing"]
                merged[i] = replay
                conflicts += 1
        if conflicts:
            with self._lock:
                self._counts["conflicts"] += conflicts
        if partial or conflicts:
            ctx.results = merged
            ctx.usage = aggregate_usage(merged)

    def prune(self) -> int:
        """Drop every expired decision now, and those over `maxsize`.

        Returns:
            How many decisions were dropped.
        """
        return self._store.prune(_now())

    def cache_info(self) -> Dict[str, Any]:
        """Counters since construction or the last `cache_clear()`.

        Returns:
            A dict with `size` (decisions held), `maxsize` (None with a custom store), `hits` and
            `misses` (per state looked up), and `conflicts`: decisions computed for a request that
            another run had already stored differently, which were replaced by the stored one.
        """
        size = len(self._store)
        with self._lock:
            return {"size": size, "maxsize": self.maxsize, **self._counts}

    def cache_clear(self) -> None:
        """Forget every stored decision and reset the counters."""
        self._store.clear()
        with self._lock:
            for name in self._counts:
                self._counts[name] = 0

    def close(self) -> None:
        """Close the store, if it holds anything open. Do not use the cache afterwards."""
        self._store.close()
