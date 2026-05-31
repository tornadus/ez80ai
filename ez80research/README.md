# ez80research — autonomous model-improvement loop

An [autoresearch](https://github.com/karpathy/autoresearch)-style agentic loop
for NEOCHAT, adapted to the Ti-84 Plus CE's hard memory budget. Claude Code is
the agent; this directory is the scaffolding (the "rules of the game").

## Files

| File | Role | Editable by agent? |
|------|------|--------------------|
| `program.md` | Research brief that steers the agent | **No** |
| `evaluate.py` | Feasibility + quality harness (IntAcc re-measure, real export+build, size gate, single `VERDICT` line) | **No** |
| `budget.py` | Re-export of top-level `sizes.py` budget helpers | **No** |
| `run_experiment.sh` | Commit → bounded train → evaluate → keep/revert → log | **No** |
| `results.tsv` | Append-only experiment log (git-untracked) | append-only |
| `baseline.json` | Frozen baseline verdict (git-untracked) | — |
| `run.log` / `eval.log` | Last train / eval stdout (git-untracked) | — |

`sizes.py` (repo root) is the canonical calculator-budget helper; it is also
imported by `buildchat84.py` so the production build fails loudly when over
budget. The harness uses the **integer** inference path and runs the **real**
export+build, so the metric cannot be gamed.

## One experiment

```bash
bash ez80research/run_experiment.sh "raise hidden width to 768"
```

This commits the working-tree edit, trains (bounded), evaluates, then either
keeps the commit (if `pass=1` and IntAcc beat the best by `MARGIN`) or
`git reset --hard` back, and appends a row to `results.tsv`.

Env knobs: `EPOCHS` (default 120), `TRAIN_TIMEOUT` (600s), `EVAL_SAMPLES`
(8000), `MARGIN` (0.005), `DATA` (training_data.txt).

## Establish the baseline first

```bash
bash ez80research/run_experiment.sh "baseline"            # on a clean tree
# then record it:
python3 - <<'PY'
import json, re
v = [l for l in open('ez80research/eval.log') if l.startswith('VERDICT')][-1]
g = lambda k: re.search(rf'{k}=([^\s]+)', v).group(1)
json.dump({'intacc': float(g('intacc')), 'ram_kb': float(g('ram_kb')),
           'n_appvars': int(g('appvars')), 'maxav_kb': float(g('maxav_kb'))},
          open('ez80research/baseline.json','w'), indent=2)
print('baseline.json written')
PY
```

Every later experiment must beat `baseline.json`'s IntAcc.

## Run the loop (Claude Code)

`program.md` IS the operating manual — it contains the full loop, the
subagent-per-iteration pattern, and the stop policy. To launch a fresh instance,
just point it there:

> Read `ez80research/program.md` and run the loop it describes. Begin.

The instance acts as the strategist: it forms one hypothesis at a time and runs
each experiment in a **fresh subagent** (so its own context stays clean), then
reads the returned `VERDICT` and continues. It runs **indefinitely** and only
stops on an OS/hardware failure that prevents progress — a failed experiment is
a normal outcome, not a stop condition. `results.tsv` + git history are its
durable memory across context resets.

## Stop / reset

- Stop: interrupt the `/goal`. In-flight experiments finish atomically (one
  commit, kept or reverted).
- Inspect: `column -t -s$'\t' ez80research/results.tsv`
- Reset to baseline: `git reset --hard <baseline commit>` (kept experiments are
  ordinary commits on the branch).
