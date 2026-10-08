package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.hooks.AsyncHook;
import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** {@link AsyncHook} runs a hook elsewhere but still waits for it, as the reference does. */
class AsyncHookTest {

    private static PredictContext ctx() {
        return new PredictContext(List.of("state"), Map.of(), "english", null);
    }

    @Test
    @DisplayName("the callback runs on the executor, not the calling thread")
    void runsOnTheExecutor() {
        ExecutorService pool = Executors.newSingleThreadExecutor(r -> new Thread(r, "hook-pool"));
        try {
            AtomicReference<String> ran = new AtomicReference<>();
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    ran.set(Thread.currentThread().getName());
                }
            }, pool);
            async.onPredictStart(ctx());
            assertEquals("hook-pool", ran.get());
            assertNotEquals(Thread.currentThread().getName(), ran.get());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("the call waits: the callback has finished before the method returns")
    void waitsForTheCallback() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            CountDownLatch done = new CountDownLatch(1);
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    try {
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    done.countDown();
                }
            }, pool);
            async.onPredictStart(ctx());
            assertEquals(0, done.getCount(), "returned before the callback finished");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("the delegate's exception reaches the caller unwrapped, so the policy sees it")
    void propagatesTheDelegateFailure() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            IllegalStateException boom = new IllegalStateException("from the hook");
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictEnd(PredictContext c) {
                    throw boom;
                }
            }, pool);
            assertSame(boom, assertThrows(IllegalStateException.class,
                    () -> async.onPredictEnd(ctx())));
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("a callback that outlasts the timeout fails the dispatch")
    void timesOut() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onRoute(PredictContext c) {
                    try {
                        Thread.sleep(5_000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }, pool, Duration.ofMillis(50));
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, () -> async.onRoute(ctx()));
            assertTrue(thrown.getMessage().contains("did not finish"), thrown.getMessage());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("an overrunning callback is abandoned, so its late writes are refused")
    void anOverrunningCallbackIsAbandoned() throws Exception {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            CountDownLatch wrote = new CountDownLatch(1);
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    try {
                        Thread.sleep(400);          // overruns the 50ms deadline below
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    try {
                        c.states(List.of("written after the call moved on"));
                    } catch (RuntimeException refused) {
                        wrote.countDown();          // refused, which is the point
                    }
                }
            }, pool, Duration.ofMillis(50));
            PredictContext ctx = ctx();
            assertThrows(IllegalStateException.class, () -> async.onPredictStart(ctx));
            // cancel(true) cannot interrupt a task already on an executor, so it keeps running.
            // ctx.abandon is what stops it writing to a call that has finished.
            assertTrue(wrote.await(3, TimeUnit.SECONDS), "the late write was never attempted");
            assertEquals(List.of("state"), ctx.states(),
                    "an abandoned callback's write reached the context");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("a non-positive timeout is refused rather than meaning 'never wait'")
    void refusesANonPositiveTimeout() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            Hook plain = new Hook() { };
            assertThrows(IllegalArgumentException.class,
                    () -> AsyncHook.of(plain, pool, Duration.ZERO));
            assertThrows(IllegalArgumentException.class,
                    () -> AsyncHook.of(plain, pool, Duration.ofMillis(-1)));
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("a serialized dispatch refuses an async hook before its callback can deadlock")
    void refusesAsyncHooksUnderASerialLock() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            CountDownLatch entered = new CountDownLatch(1);
            Hooks.Policy serial = Hooks.Policy.raising().lock(new ReentrantLock());
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    entered.countDown();
                    Hooks.dispatch(List.of(new Hook() { }), Hooks.Event.PREDICT_START, c, serial);
                }
            }, pool, Duration.ofMillis(100));
            IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                    () -> Hooks.dispatch(List.of(async), Hooks.Event.PREDICT_START, ctx(), serial));
            assertTrue(thrown.getMessage().contains("concurrent(true)"), thrown.getMessage());
            assertEquals(1, entered.getCount(), "the callback ran before the unsafe policy was refused");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("every event is forwarded, not just the predict pair")
    void forwardsEveryEvent() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            List<String> seen = new ArrayList<>();
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    seen.add("start");
                }

                @Override
                public void onPredictEnd(PredictContext c) {
                    seen.add("end");
                }

                @Override
                public void onRoute(PredictContext c) {
                    seen.add("route");
                }

                @Override
                public void onLoad(PredictContext c) {
                    seen.add("load");
                }

                @Override
                public void onEvict(PredictContext c) {
                    seen.add("evict");
                }

                @Override
                public void onError(PredictContext c) {
                    seen.add("error");
                }
            }, pool);
            PredictContext ctx = ctx();
            async.onPredictStart(ctx);
            async.onRoute(ctx);
            async.onLoad(ctx);
            async.onEvict(ctx);
            async.onError(ctx);
            async.onPredictEnd(ctx);
            assertEquals(List.of("start", "route", "load", "evict", "error", "end"), seen);
        } finally {
            pool.shutdownNow();
        }
    }
}
