package com.convaiinnovations.laya;

import java.util.List;
import java.util.Map;

/**
 * Anything that can answer one question set over many states in a shared forward pass.
 *
 * <p>Separate from {@link Predictor} rather than folded into it, because not every runtime
 * batches. {@link Agent} does; {@link Router} does not — it selects a checkpoint per state, so
 * there is no single graph call to share. The reference draws the same line at runtime, raising
 * {@code TypeError} when a runner has no {@code predict_batch} rather than looping {@code decide}
 * N times behind the caller's back; here the line is drawn by the type, so the same mistake does
 * not compile.
 *
 * <p>{@link Agent#predictBatch(List, Map)} already had this signature, so this interface names an
 * existing capability rather than adding one.
 */
public interface BatchPredictor {

    /**
     * Ask every question about every state.
     *
     * @return one prediction per state, <b>in the order the states were given</b>, which is the
     *     property a caller zipping the results back onto its own inputs depends on
     */
    List<Prediction> predictBatch(List<?> states, Map<String, Question> questions);
}
