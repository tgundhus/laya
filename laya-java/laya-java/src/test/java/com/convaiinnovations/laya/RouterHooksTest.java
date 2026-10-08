package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Router.Checkpoint;
import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A Router-level hook sees what no agent can: which checkpoint was chosen, and when one was built
 * or dropped. Before this, {@code Router} dispatched nothing at all.
 */
class RouterHooksTest {

    /** Records every event with the model it carried. */
    private static final class Trace implements Hook {

        final List<String> events = new ArrayList<>();

        private void note(String event, PredictContext ctx) {
            events.add(event + ":" + ctx.model());
        }

        @Override
        public void onRoute(PredictContext ctx) {
            note("route", ctx);
        }

        @Override
        public void onLoad(PredictContext ctx) {
            note("load", ctx);
        }

        @Override
        public void onEvict(PredictContext ctx) {
            note("evict", ctx);
        }

        @Override
        public void onPredictStart(PredictContext ctx) {
            note("start", ctx);
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            note("end", ctx);
        }
    }

    private static final class StubAgents implements Router.AgentFactory {

        private final Path root;
        int builds;

        StubAgents(Path root) {
            this.root = root;
        }

        @Override
        public Agent create(Checkpoint checkpoint) throws IOException {
            builds++;
            return TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession());
        }
    }

    private static Map<String, Question> questions() {
        return Map.of("urgent", Question.noul("Needs a human."));
    }

    @Test
    @DisplayName("route() dispatches on_route once, naming the checkpoint it chose")
    void routeDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.hooks().addHook(trace);
            router.route("hello", questions());
            assertEquals(1, trace.events.size(), trace.events.toString());
            assertTrue(trace.events.get(0).startsWith("route:"), trace.events.toString());
        }
    }

    @Test
    @DisplayName("load() dispatches on_load for a build and stays quiet for a cache hit")
    void loadDispatchesOnlyOnBuild(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        StubAgents agents = new StubAgents(root);
        try (Router router = Router.builder().agents(agents).maxLoaded(2).build()) {
            router.hooks().addHook(trace);
            router.load("english");
            assertEquals(List.of("load:english"), trace.events);
            router.load("english");                    // resident: no second build, no second event
            assertEquals(List.of("load:english"), trace.events);
            assertEquals(1, agents.builds);
        }
    }

    @Test
    @DisplayName("an eviction forced by maxLoaded dispatches on_evict for the victim")
    void evictionDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(1).build()) {
            router.hooks().addHook(trace);
            router.load("english");
            router.load("multilingual");               // maxLoaded 1, so english must go
            assertTrue(trace.events.contains("evict:english"), trace.events.toString());
            assertEquals(List.of("load:english", "load:multilingual", "evict:english"),
                    trace.events);
        }
    }

    @Test
    @DisplayName("unload dispatches on_evict for what it freed")
    void unloadDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.load("english");
            router.hooks().addHook(trace);             // installed after the load
            assertEquals(List.of(Checkpoint.ENGLISH), router.unload("english"));
            assertEquals(List.of("evict:english"), trace.events);
        }
    }

    @Test
    @DisplayName("predict dispatches one start/end pair for the whole route-and-answer call")
    void predictDispatchesOnePair(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.hooks().addHook(trace);
            router.predict("hello", questions());
            assertEquals(1, trace.events.stream().filter(e -> e.startsWith("start:")).count(),
                    trace.events.toString());
            assertEquals(1, trace.events.stream().filter(e -> e.startsWith("end:")).count(),
                    trace.events.toString());
            // on_route first, then on_load inside the predict pair -- the reference's order.
            // Derived from the trace, not hard-coded: which checkpoint answers is the default's
            // business, and this test is about the sequence.
            List<String> kinds = trace.events.stream().map(e -> e.split(":")[0]).toList();
            assertEquals(List.of("route", "start", "load", "end"), kinds,
                    trace.events.toString());
        }
    }

    @Test
    @DisplayName("on_load is dispatched with no router lock held, as seen from a SECOND thread")
    void dispatchHoldsNoRouterLock(@TempDir Path root) throws Exception {
        TinyCheckpoint.write(root, 64, 32);
        // A second thread, not the dispatching one: `lock` is a ReentrantLock, so a callback on the
        // dispatching thread re-enters it successfully whether or not it is held. Only another
        // thread can tell the difference.
        AtomicBoolean reached = new AtomicBoolean();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(2).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onLoad(PredictContext ctx) {
                    Thread other = new Thread(() -> {
                        router.loaded();
                        reached.set(true);
                    }, "router-lock-probe");
                    other.start();
                    try {
                        other.join(5_000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            });
            router.load("english");
            assertTrue(reached.get(),
                    "a second thread could not reach the router during on_load, so the dispatch "
                    + "held the lock");
        }
    }

    @Test
    @DisplayName("unloadAll dispatches on_evict for every checkpoint, which is what close() uses")
    void unloadAllDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(3).build()) {
            router.load("english");
            router.load("multilingual");
            router.hooks().addHook(trace);
            assertEquals(2, router.unloadAll().size());
            assertEquals(List.of("evict:english", "evict:multilingual"), trace.events);
        }
    }

    @Test
    @DisplayName("a failed eviction hook still receives every checkpoint freed by close")
    void closeDispatchesEveryEvictionAfterAHookFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(3).build();
        router.preload(List.of("english", "multilingual"));
        List<String> evicted = new ArrayList<>();
        IllegalStateException failure = new IllegalStateException("sink unavailable");
        router.hooks().addHook(new Hook() {
            @Override
            public void onEvict(PredictContext ctx) {
                evicted.add(ctx.model());
                throw failure;
            }
        });
        assertEquals(failure, assertThrows(IllegalStateException.class, router::close));
        assertEquals(List.of("english", "multilingual"), evicted);
        assertEquals(List.of(), router.loaded());
        router.close();
    }

    @Test
    @DisplayName("preload dispatches on_load for what it built")
    void preloadDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(3).build()) {
            router.hooks().addHook(trace);
            router.preload(List.of("english", "multilingual"));
            assertEquals(List.of("load:english", "load:multilingual"), trace.events);
        }
    }

    @Test
    @DisplayName("a throwing on_load hook does not strand the lease that lease() took")
    void aThrowingHookDoesNotLeakTheLease(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(2).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onLoad(PredictContext ctx) {
                    throw new IllegalStateException("the sink refused it");
                }
            });
            assertThrows(IllegalStateException.class, () -> router.lease("english"));
            // A stranded lease leaves the slot retired-but-never-closed: unload would report it
            // freed while the session stayed open.
            assertEquals(List.of(Checkpoint.ENGLISH), router.unload("english"));
        }
    }

    @Test
    @DisplayName("a router with no hooks dispatches nothing and builds no context")
    void noHooksNoDispatch(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        StubAgents agents = new StubAgents(root);
        try (Router router = Router.builder().agents(agents).build()) {
            assertFalse(router.hooks().hooks().iterator().hasNext());
            router.predict("hello", questions());      // the unhooked fast path still answers
            assertEquals(1, agents.builds);
        }
    }
}
