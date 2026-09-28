"""Fixed request sets for the benchmarks that compare answers: one JSONL, the same for every run.

Each row is `{"id", "set", "state", "questions", "gold"}`, where `gold` maps a question name to
its expected label when the set has one. Sets:

    english13   the 13 English requests of bench_cpu_fast_path.py (tickets, emails, chats; 1-10 questions)
    feishu      the 64 Chinese research/benchmarks/feishu_zh cases in both modes (128 requests);
                gold labels for the choice mode
    massive_en  the first --per-lang rows of mteb/amazon_massive_intent `en`, as research/eval
                builds them: a 20-option choice question, gold label known
    massive_ml  the same for ten other languages, which the Router sends to the multilingual checkpoint

MASSIVE needs `datasets` and the network the first time; the JSONL is then all a run reads:

    python research/scripts/request_sets.py --out requests.jsonl
    python research/scripts/request_sets.py --out requests.jsonl --per-lang 100 --languages de,ja
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FEISHU = os.path.join(ROOT, "research", "benchmarks", "feishu_zh")
EVAL = os.path.join(ROOT, "research", "eval")
LANGUAGES = ("de", "es", "fr", "ja", "zh-CN", "ar", "hi", "ru", "sw", "th")


def english13():
    sys.path.insert(0, HERE)
    from bench_cpu_fast_path import ENGLISH  # imports torch; only the request definitions are used

    return [{"id": "en13/" + name, "set": "english13", "state": state, "questions": questions, "gold": None}
            for name, state, questions in ENGLISH]


def feishu():
    sys.path.insert(0, FEISHU)
    from audit import load_cases  # verifies the frozen dataset and prompt hashes
    from prompts import requests_for

    cases, _ = load_cases()
    rows = []
    for case in cases:
        for mode, req in requests_for(case).items():
            gold = {"category": case["expected"]} if mode == "choice" else None
            rows.append({"id": "feishu/%s/%s" % (case["id"], mode), "set": "feishu", **req, "gold": gold})
    return rows


def massive(languages, per_lang, name):
    sys.path.insert(0, EVAL)
    from laya_eval import N_OPTS, SEED, build_suite, load_language

    rows = []
    for lang in languages:
        data = load_language(lang)
        labels = sorted({r["label_text"] for r in data})
        cases, gold, keys = build_suite(data, labels, per_lang, N_OPTS, SEED)
        for n, ((state, questions), g, k) in enumerate(zip(cases, gold, keys)):
            rows.append({"id": "massive/%s/%03d" % (lang, n), "set": name, "lang": lang, "state": state,
                         "questions": questions, "gold": {"intent": k[g]}})
    return rows


def build(per_lang, languages):
    return (english13() + feishu() + massive(["en"], per_lang, "massive_en")
            + massive(languages, max(1, per_lang // 4), "massive_ml"))


def load(path, sets=None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [r for r in rows if sets is None or r["set"] in sets]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-lang", type=int, default=200,
                        help="MASSIVE rows for English; each other language gets a quarter of it")
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    args = parser.parse_args()
    rows = build(args.per_lang, [x for x in args.languages.split(",") if x])
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["set"]] = counts.get(row["set"], 0) + 1
    print("wrote %d requests to %s: %s" % (len(rows), args.out, counts))


if __name__ == "__main__":
    main()
