import com.convaiinnovations.laya.Agent;
import com.convaiinnovations.laya.Answer;
import com.convaiinnovations.laya.Prediction;
import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.Router;
import com.convaiinnovations.laya.Shortlist;
import com.convaiinnovations.laya.json.Json;
import com.convaiinnovations.laya.json.PythonJson;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JavaUpstreamSmoke {
    /** Called by java_upstream_smoke.py: checkpoint, graph directory, reference JSON, output JSON. */
    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        Map<String,Object> reference;
        try (var reader = Files.newBufferedReader(Path.of(args[2]))) {
            reference = (Map<String,Object>) Json.parse(reader);
        }
        Map<String,Question> questions = new LinkedHashMap<>();
        for (var entry : ((Map<String,Object>) reference.get("questions")).entrySet()) {
            var q = (Map<String,Object>) entry.getValue();
            var instruction = (String) q.get("instructions");
            var question = switch ((String) q.get("type")) {
                case "choice" -> Question.choice(instruction, (Map<String,Object>) q.get("criteria"));
                case "score" -> Question.score(instruction, (List<Object>) q.get("criteria"));
                default -> Question.noul(instruction);
            };
            questions.put(entry.getKey(),question);
        }
        var states = (List<Object>) reference.get("states");
        List<Object> singles = new ArrayList<>();
        List<Object> batch = new ArrayList<>();
        List<Object> oneOption = new ArrayList<>();
        List<Object> routerSingles = new ArrayList<>();
        List<Object> routerBatch = new ArrayList<>();
        Map<String,Object> tournament;
        Map<String,Object> benchmark = new LinkedHashMap<>();
        Path checkpoint = Path.of(args[0]);
        Path graph = Path.of(args[1]);
        try (var agent = Agent.open(checkpoint,graph,2);
                var router = Router.builder().agents(cp -> {
                    if (cp != Router.Checkpoint.ENGLISH) throw new IllegalArgumentException("English smoke");
                    return Agent.open(checkpoint,graph,2);
                }).build()) {
            for (var state : states) singles.add(output(agent.predict(state,questions)));
            for (var prediction : agent.predictBatch(states,questions,null,2,false)) batch.add(output(prediction));
            Map<String,Question> one = new LinkedHashMap<>();
            one.put("topic",Question.choiceOf("What is this about?",Map.entry("billing","a billing issue")));
            one.put("urgency",Question.score("How urgent is this?",List.of("urgent")));
            for (var prediction : agent.predictBatch(states,one,null,2,false)) oneOption.add(output(prediction));
            var options = Router.RouteOptions.none().model("english");
            for (var state : states) routerSingles.add(output(router.predict(state,questions,options)));
            for (var prediction : router.predictBatch(states,questions,options)) routerBatch.add(output(prediction));
            Map<String,Object> labels = new LinkedHashMap<>();
            for (String label : List.of("billing","refunds","fraud","cards","transfers","loans",
                    "savings","kyc","support","legal","sales","tech","identity","fees",
                    "cash","statements","login","cancel","lost","other")) labels.put(label,label);
            Map<String,Question> wide = Map.of("intent",Question.choice("What does the customer need?",labels));
            var narrowed = Shortlist.predictTournament(agent,states.get(0),wide,8);
            var bracket = narrowed.tournament().get("intent");
            tournament = new LinkedHashMap<>();
            tournament.put("prediction",output(narrowed.prediction()));
            tournament.put("labels",bracket.labels());
            tournament.put("total",bracket.total());
            tournament.put("rounds",bracket.rounds());
            List<Double> sequentialMs = new ArrayList<>();
            List<Double> batchedMs = new ArrayList<>();
            List<String> order = new ArrayList<>();
            for (int repeat = 0; repeat < 5; repeat++) {
                List<Object> sequential = new ArrayList<>();
                List<Object> batched = new ArrayList<>();
                double sequentialTime = 0;
                double batchedTime = 0;
                boolean sequentialFirst = repeat % 2 == 0;
                for (int step = 0; step < 2; step++) {
                    boolean runSequential = step == 0 ? sequentialFirst : !sequentialFirst;
                    long started = System.nanoTime();
                    if (runSequential) {
                        for (var state : states) sequential.add(output(router.predict(state,questions,options)));
                        sequentialTime = (System.nanoTime()-started)/1e6;
                    } else {
                        for (var prediction : router.predictBatch(states,questions,options)) {
                            batched.add(output(prediction));
                        }
                        batchedTime = (System.nanoTime()-started)/1e6;
                    }
                }
                if (!sequential.equals(routerSingles) || !batched.equals(routerBatch)) {
                    throw new IllegalStateException("a benchmark repeat changed an answer or its usage");
                }
                if (repeat >= 2) {
                    sequentialMs.add(sequentialTime);
                    batchedMs.add(batchedTime);
                    order.add(sequentialFirst ? "sequential,batch" : "batch,sequential");
                }
            }
            benchmark.put("warmup_repeats_per_mode",2);
            benchmark.put("measured_repeats_per_mode",3);
            benchmark.put("requests_per_repeat",states.size());
            benchmark.put("ort_threads",2);
            benchmark.put("measured_mode_order",order);
            benchmark.put("sequential_whole_call_ms",sequentialMs);
            benchmark.put("batch_whole_call_ms",batchedMs);
            benchmark.put("repeat_answers_and_usage_unchanged",true);
        }
        Map<String,Object> report = new LinkedHashMap<>();
        report.put("sdk","java");
        report.put("graph",Files.isRegularFile(graph.resolve("laya.onnx")) ? "fused" : "split");
        report.put("singles",singles);
        report.put("batch",batch);
        report.put("one_option",oneOption);
        report.put("router_singles",routerSingles);
        report.put("router_batch",routerBatch);
        report.put("tournament",tournament);
        report.put("benchmark",benchmark);
        Files.writeString(Path.of(args[3]),PythonJson.dumps(report));
        System.out.println("Java checkpoint smoke complete");
    }

    private static Map<String,Object> output(Prediction prediction) {
        Map<String,Object> answers = new LinkedHashMap<>();
        for (var entry : prediction.answers().entrySet()) {
            var answer = entry.getValue();
            Map<String,Object> value = new LinkedHashMap<>();
            value.put("type",answer.type());
            value.put("confidence",answer.confidence());
            value.put("answer_confidence",answer.answerConfidence());
            value.put("action",Map.of("act_probability",answer.actProbability()));
            if (answer instanceof Answer.Choice choice) {
                value.put("choice",choice.choice());
                value.put("probabilities",choice.probabilities());
            } else if (answer instanceof Answer.Score score) {
                value.put("score",score.score());
                value.put("legend",score.legend());
                value.put("probabilities",score.probabilities());
            } else if (answer instanceof Answer.Noul noul) {
                value.put("noul",noul.noul());
            }
            answers.put(entry.getKey(),value);
        }
        var usage = prediction.usage();
        Map<String,Object> used = new LinkedHashMap<>();
        used.put("input_tokens",usage.inputTokens());
        used.put("output_tokens",usage.outputTokens());
        used.put("state_tokens",usage.stateTokens());
        used.put("state_tokens_dropped",usage.stateTokensDropped());
        used.put("truncated",usage.truncated());
        used.put("truncated_questions",usage.truncatedQuestions());
        Map<String,Object> result = new LinkedHashMap<>();
        result.put("answers",answers);
        result.put("usage",used);
        return result;
    }
}
