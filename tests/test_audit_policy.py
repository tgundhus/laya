"""The dependency audit's policy: python tests/test_audit_policy.py (no network, no weights).

`scripts/pip_audit_policy.py` decides whether the audit in .github/workflows/security.yml passes.
It may pass an advisory only while that advisory is accepted, unfixed and on the package it was
accepted for; anything else fails the audit, as does a dependency pip-audit could not audit.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pip_audit_policy as policy  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10; the audit job, which reads the list, runs 3.11
    tomllib = None

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: raised %r, expected %s" % (name, e, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))


def report(*deps):
    return {"dependencies": list(deps), "fixes": []}


def dep(name, version, *vulns):
    return {"name": name, "version": version, "vulns": list(vulns)}


def vuln(vid, aliases=(), fixes=()):
    return {"id": vid, "aliases": list(aliases), "fix_versions": list(fixes), "description": ""}


ENTRY = {"id": "PYSEC-1", "aliases": ["CVE-1", "GHSA-1"], "package": "chromadb", "reason": "server only"}
ACCEPTED = {i: ENTRY for i in ["PYSEC-1", "CVE-1", "GHSA-1"]}

# --------------------------------------------------------------- verdicts
fails, warns, stale = policy.evaluate([report(dep("chromadb", "1.1.1", vuln("PYSEC-1")))], ACCEPTED)
check("accepted and unfixed passes", fails, [])
check("accepted and unfixed stays visible as a warning", len(warns), 1)
check("an accepted entry that was seen is not stale", stale, [])

fails, warns, _ = policy.evaluate([report(dep("chromadb", "1.1.1", vuln("GHSA-1")))], ACCEPTED)
check("an alias matches the accepted entry", (fails, len(warns)), ([], 1))

fails, warns, _ = policy.evaluate([report(dep("chromadb", "1.1.1", vuln("PYSEC-1", fixes=["1.6.0"])))], ACCEPTED)
check("an accepted advisory fails once a fix exists", len(fails), 1)
check("that failure names the fixed release", "1.6.0" in fails[0], True)

fails, _, _ = policy.evaluate([report(dep("json-repair", "0.25.2", vuln("GHSA-2", fixes=["0.60.1"])))], ACCEPTED)
check("an advisory nobody accepted fails", len(fails), 1)
fails, _, _ = policy.evaluate([report(dep("nltk", "3.10.3", vuln("PYSEC-9")))], ACCEPTED)
check("an unaccepted advisory with no fix fails too", len(fails), 1)

fails, _, _ = policy.evaluate([report(dep("somethingelse", "1.0", vuln("PYSEC-1")))], ACCEPTED)
check("an entry accepted for one package does not pass another", len(fails), 1)

fails, _, _ = policy.evaluate([report({"name": "torch", "skip_reason": "not on PyPI"})], ACCEPTED)
check("a dependency that could not be audited fails (--strict)", len(fails), 1)

twice = report(dep("chromadb", "1.1.1", vuln("PYSEC-1"), vuln("PYSEC-1")))
fails, warns, _ = policy.evaluate([twice, twice], ACCEPTED)
check("an advisory reported twice, or by two groups, counts once", (fails, len(warns)), ([], 1))

fails, warns, stale = policy.evaluate([report(dep("chromadb", "1.1.1"))], ACCEPTED)
check("an accepted entry no report mentions is stale", (fails, warns, stale), ([], [], ["PYSEC-1"]))

fails, _, _ = policy.evaluate([report(dep("Json_Repair", "0.25.2", vuln("PYSEC-3")))],
                              {"PYSEC-3": dict(ENTRY, id="PYSEC-3", package="json-repair")})
check("package names compare as pip normalises them", fails, [])

# --------------------------------------------------------------- the accepted list
# Reading TOML needs Python 3.11. The verdicts above run everywhere; the list and the command are
# checked on the 3.11+ lanes, and the audit job itself runs 3.11.
if tomllib is None:
    print("Python %d.%d has no tomllib: the accepted list and the command are checked on 3.11+"
          % sys.version_info[:2])
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    sys.exit(1 if FAIL else 0)

with tempfile.TemporaryDirectory() as tmp:
    bad = os.path.join(tmp, "bad.toml")
    with open(bad, "w") as fh:
        fh.write('[[advisory]]\nid = "PYSEC-4"\npackage = "x"\n')
    check_raises("an entry without a reason is refused", ValueError, lambda: policy.load_accepted(bad))

LIST = os.path.join(ROOT, ".github", "accepted-advisories.toml")
with open(LIST, "rb") as fh:
    entries = tomllib.load(fh)["advisory"]
check("the accepted list is not empty", len(entries) > 0, True)
FIELDS = ("id", "aliases", "package", "via", "reason", "reviewed")
check("every entry says what, where from, why and when",
      [e["id"] for e in entries if not all(e.get(k) for k in FIELDS)], [])
ids = [i for e in entries for i in [e["id"], *e["aliases"]]]
check("no id is accepted twice", len(ids), len(set(ids)))
check("the list loads", len(policy.load_accepted(LIST)), len(ids))

# --------------------------------------------------------------- the command and its wiring
with tempfile.TemporaryDirectory() as tmp:
    ok, bad = os.path.join(tmp, "ok.json"), os.path.join(tmp, "bad.json")
    with open(ok, "w") as fh:
        json.dump(report(dep("chromadb", "1.1.1", vuln("PYSEC-2026-311"))), fh)
    with open(bad, "w") as fh:
        json.dump(report(dep("json-repair", "0.25.2", vuln("GHSA-xf7x-x43h-rpqh", fixes=["0.60.1"]))), fh)
    devnull = open(os.devnull, "w")
    stdout, sys.stdout = sys.stdout, devnull
    try:
        codes = (policy.main(["--accepted", LIST, ok]), policy.main(["--accepted", LIST, ok, bad]))
    finally:
        sys.stdout = stdout
        devnull.close()
check("the command passes an accepted report and fails one with a fixable advisory", codes, (0, 1))

with open(os.path.join(ROOT, ".github", "workflows", "security.yml"), encoding="utf-8") as fh:
    workflow = fh.read()
check("the audit runs the policy on the accepted list",
      "scripts/pip_audit_policy.py --accepted .github/accepted-advisories.toml" in workflow, True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
