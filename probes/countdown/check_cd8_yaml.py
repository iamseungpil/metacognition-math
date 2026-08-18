#!/usr/bin/env python
"""Self-audit for countdown_rl_8arm.yaml. Prints counts, never guesses."""
import difflib
import re
import sys

import yaml

TGT = "/home/v-seungplee/metacognition-math/countdown_rl_8arm.yaml"
REF = "/home/v-seungplee/metacognition-math/h100std_rq3v2g_b4p3g.yaml"

fails = []


def note(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


raw = open(TGT).read()
lines = raw.split("\n")

# ── 1. parses as YAML ────────────────────────────────────────────────────────
doc = yaml.safe_load(raw)
jobs = doc["jobs"]
note(len(jobs) == 9, "job count = %d (expect 9: gs0 + A..H)" % len(jobs))
print("     job names:", [j["name"] for j in jobs])

# ── 2. single-quote accounting, per job command and vs the reference ─────────
ref_jobs = yaml.safe_load(open(REF).read())["jobs"]
ref_cmd = ref_jobs[0]["command"][1]
ref_sq = ref_cmd.count("'")
print("     reference launcher (b4p3g) single quotes in command body: %d" % ref_sq)

ref_wrap = 2
ref_in_comments = ref_sq - ref_wrap
print("     reference breakdown: %d wrapper + %d inside comments (balanced pairs)"
      % (ref_wrap, ref_in_comments))

tot = 0
for j in jobs:
    body = j["command"][1]
    n = body.count("'")
    tot += n
    note(n % 2 == 0 and n == 2,
         "job %-10s command single-quote count = %d (even, wrapper pair only; reference %d = 2 wrapper + %d in comments)"
         % (j["name"], n, ref_sq, ref_in_comments))
    # the two quotes must be the wrapper: first is `bash -c '`, last closes it
    stripped = body.strip()
    note(stripped.startswith("bash -c '") and stripped.endswith("'"),
         "job %-10s quotes are the bash -c wrapper pair (opens with bash -c ' and ends with ')" % j["name"])

note(raw.count("'") == tot, "file-level single quotes = %d, all of them inside command bodies (%d)"
     % (raw.count("'"), tot))

# ── 3. backslash-continuation hygiene ────────────────────────────────────────
bad = []
for i, ln in enumerate(lines[:-1]):
    if ln.rstrip().endswith("\\"):
        nxt = lines[i + 1].strip()
        if nxt == "" or nxt.startswith("#"):
            bad.append((i + 1, ln.strip()[:60], nxt[:40]))
cont = sum(1 for ln in lines if ln.rstrip().endswith("\\"))
note(not bad, "backslash-continued lines = %d, none followed by a comment or blank line %s"
     % (cont, bad if bad else ""))

# ── 4. amlt runtime escaping: no bare ${...} or $( ) inside bash -c bodies ───
for j in jobs:
    body = j["command"][1]
    bare_var = [m.group(0) for m in re.finditer(r"(?<!\$)\$\{[A-Za-z_]", body)]
    bare_sub = [m.group(0) for m in re.finditer(r"(?<!\$)\$\((?!\()", body)]
    note(not bare_var and not bare_sub,
         "job %-10s no bare ${} / $() at runtime (bare_var=%d bare_sub=%d)"
         % (j["name"], len(bare_var), len(bare_sub)))

# ── 5. the eight arm bodies must be byte-identical ──────────────────────────
arms = [j for j in jobs if j["name"] != "cd8_gs0"]
base = arms[0]["command"][1]
for j in arms[1:]:
    note(j["command"][1] == base, "arm %-10s command body byte-identical to arm A" % j["name"])

# arm env blocks differ ONLY by ARM
base_env = dict(arms[0]["submit_args"]["env"])
for j in arms[1:]:
    e = dict(j["submit_args"]["env"])
    diff = {k for k in set(e) | set(base_env) if e.get(k) != base_env.get(k)}
    note(diff == {"ARM"}, "arm %-10s env differs from arm A only in %s" % (j["name"], sorted(diff)))

print("     ARM values:", [j["submit_args"]["env"]["ARM"] for j in arms])

# ── 6. G8 textual diff (the readability claim) ──────────────────────────────
def block(name):
    out, on = [], False
    for ln in lines:
        if ln.startswith("  - name: "):
            on = ln.strip() == "- name: " + name
        elif on and ln.startswith("  # "):
            on = False  # the banner comment belongs to the NEXT job
        if on:
            out.append(ln)
    return out

a, g = block("cd8_corr"), block("cd8_neg")
d = [x for x in difflib.unified_diff(a, g, lineterm="", n=0) if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))]
note(len(d) == 4, "G8 diff arm A vs arm G = %d changed lines (expect 4: name x2, ARM x2)" % len(d))
for x in d:
    print("       ", x)

# ── 7. no reward weight leaked into the yaml ────────────────────────────────
leak = [ln.strip() for ln in lines
        if re.search(r"(w_meta|w_gate|w_corr|w_format|meta_floor|reversal|shift|w_over|w_cal|len_cost)\s*=", ln)]
note(not leak, "no reward weight assignment in the yaml (%d found) %s" % (len(leak), leak[:3]))

# ── 8. the single treatment key is present exactly once per arm ─────────────
for j in arms:
    n = j["command"][1].count("++algorithm.countdown_arm=")
    note(n == 1, "arm %-10s carries ++algorithm.countdown_arm exactly once (%d)" % (j["name"], n))

# ── 9. staging guards present ───────────────────────────────────────────────
for j in jobs:
    b = j["command"][1]
    n_exit = b.count("ABORT window")
    note(n_exit >= 5, "job %-10s fail-closed aborts = %d" % (j["name"], n_exit))

# ── 10. post-substitution bash syntax check (amlt turns $$ into $) ──────────
import os
import subprocess
import tempfile

for j in jobs:
    body = j["command"][1]
    runtime = body.replace("$$", "$")
    fd, path = tempfile.mkstemp(suffix=".sh")
    with os.fdopen(fd, "w") as fh:
        fh.write(runtime)
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    note(r.returncode == 0, "job %-10s post-substitution body passes bash -n %s"
         % (j["name"], r.stderr.strip()[:200]))
    os.unlink(path)

print("\n%s  (%d failures)" % ("ALL CHECKS PASS" if not fails else "CHECKS FAILED", len(fails)))
sys.exit(1 if fails else 0)
