# Prediction hook examples

Small, runnable hooks. See [`docs/hooks/index.md`](../../docs/hooks/index.md) for the full reference.

- [`audit.py`](audit.py): log every decision and optionally ship it to an external service.
- [`redact.py`](redact.py): strip emails/phones from the state before inference.
- [`cache.py`](cache.py): cache decisions and skip the forward pass on a hit.
- [`decision_store_redis.py`](decision_store_redis.py): a `DecisionStore` over Redis, so several machines replay one set of `DecisionCache` decisions. Its self-check needs Redis or fakeredis, not a checkpoint.
- [`otel.py`](otel.py): counters and a latency histogram per decision.

Each example loads a real checkpoint, so the first run downloads it.
