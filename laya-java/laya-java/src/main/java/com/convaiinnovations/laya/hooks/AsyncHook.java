package com.convaiinnovations.laya.hooks;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * Runs another hook's callbacks on an {@link Executor}, and waits for each one.
 *
 * <p>For a hook that must run on a particular thread -- a framework request scope, a single-threaded
 * actor, a UI loop -- not for making a slow hook free. The reference does the same: an async hook's
 * coroutine is driven to completion on a background loop ({@code laya/hooks.run_coroutine_sync}), so
 * the prediction still waits for it.
 *
 * <p>Waiting is what keeps a thrown exception reaching the hook policy, and keeps the mutable
 * {@link PredictContext} from being read after the call moved on.
 *
 * <p>Except on a deadline. {@code CompletableFuture.cancel} does not interrupt a task already
 * running on an executor, so an overrunning callback keeps going. It is {@link
 * PredictContext#abandon abandoned} instead, exactly as {@link Hooks} does for a hook that
 * overruns {@code hooks_timeout}: anything it writes to the context afterwards is refused. It still
 * occupies the executor's thread until it returns, so a single-threaded executor has none left for
 * the next callback -- size the executor for the deadline, or leave the deadline off.
 */
public final class AsyncHook implements Hook {

    private final Hook delegate;
    private final Executor executor;
    private final Duration timeout;

    private AsyncHook(Hook delegate, Executor executor, Duration timeout) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.executor = Objects.requireNonNull(executor, "executor");
        this.timeout = timeout;
    }

    /** Runs {@code delegate} on {@code executor}, waiting indefinitely for each callback. */
    public static AsyncHook of(Hook delegate, Executor executor) {
        return new AsyncHook(delegate, executor, null);
    }

    /**
     * Runs {@code delegate} on {@code executor}, failing a callback that outlasts {@code timeout}.
     *
     * <p>Independent of {@code hooks_timeout}: that one bounds the whole dispatch and is not visible
     * to a hook, so this one has to be given here.
     */
    public static AsyncHook of(Hook delegate, Executor executor, Duration timeout) {
        if (timeout != null && (timeout.isNegative() || timeout.isZero())) {
            throw new IllegalArgumentException("timeout must be positive, got " + timeout);
        }
        return new AsyncHook(delegate, executor, timeout);
    }

    /** The hook whose callbacks this runs. */
    public Hook delegate() {
        return delegate;
    }

    @Override
    public void onPredictStart(PredictContext ctx) {
        await("on_predict_start", ctx, delegate::onPredictStart);
    }

    @Override
    public void onPredictEnd(PredictContext ctx) {
        await("on_predict_end", ctx, delegate::onPredictEnd);
    }

    @Override
    public void onRoute(PredictContext ctx) {
        await("on_route", ctx, delegate::onRoute);
    }

    @Override
    public void onLoad(PredictContext ctx) {
        await("on_load", ctx, delegate::onLoad);
    }

    @Override
    public void onEvict(PredictContext ctx) {
        await("on_evict", ctx, delegate::onEvict);
    }

    @Override
    public void onError(PredictContext ctx) {
        await("on_error", ctx, delegate::onError);
    }

    private void await(String event, PredictContext ctx, Consumer<PredictContext> callback) {
        // The executor picks the thread, so the task reports it back: abandoning the context needs
        // the thread that will be writing to it.
        AtomicReference<Thread> runner = new AtomicReference<>();
        CompletableFuture<Void> running;
        try {
            running = CompletableFuture.runAsync(() -> {
                runner.set(Thread.currentThread());
                callback.accept(ctx);
            }, executor);
        } catch (RuntimeException rejected) {
            // A saturated or shut-down executor. Thrown as-is so the hook policy decides, exactly
            // as it would for a synchronous hook that threw.
            throw rejected;
        }
        try {
            if (timeout == null) {
                running.join();
            } else {
                running.get(timeout.toNanos(), TimeUnit.NANOSECONDS);
            }
        } catch (CompletionException | java.util.concurrent.ExecutionException wrapper) {
            // Unwrapped, so the caller sees what the delegate threw rather than the plumbing.
            Throwable cause = wrapper.getCause() == null ? wrapper : wrapper.getCause();
            if (cause instanceof RuntimeException problem) {
                throw problem;
            }
            if (cause instanceof Error problem) {
                throw problem;
            }
            throw new IllegalStateException(delegate + " failed in " + event, cause);
        } catch (TimeoutException expired) {
            // cancel(true) cannot interrupt a task already running on an executor, so the callback
            // is cut off from the call instead: whatever it writes from here on is refused.
            Thread overrunning = runner.get();
            if (overrunning != null) {
                ctx.abandon(overrunning);
            }
            running.cancel(true);
            throw new IllegalStateException(
                    delegate + " did not finish " + event + " within " + timeout, expired);
        } catch (InterruptedException interrupted) {
            // The flag is restored before unwinding, so a caller using interruption to cancel is
            // not left thinking the interrupt was swallowed.
            Thread.currentThread().interrupt();
            running.cancel(true);
            throw new IllegalStateException(delegate + " was interrupted in " + event, interrupted);
        }
    }

    @Override
    public String toString() {
        return "AsyncHook(" + delegate + (timeout == null ? "" : ", timeout=" + timeout) + ")";
    }
}
