#!/usr/bin/env bash
# One ez80research experiment: commit the edit, train (bounded), evaluate via
# the feasibility harness, decide keep/revert greedily, and log one row.
#
# Usage:  bash ez80research/run_experiment.sh "<short hypothesis label>"
#
# Env knobs:
#   EPOCHS        training epochs (fairness unit; default 120)
#   TRAIN_TIMEOUT wall-clock safety ceiling in seconds (default 600)
#   EVAL_SAMPLES  harness IntAcc sample count (default 8000)
#   MARGIN        min IntAcc improvement to keep, guards noise (default 0.005)
#   DATA          training data file (default training_data.txt)
#
# The script makes the keep/revert decision itself and never asks for
# confirmation. Each experiment is exactly one git commit (kept or reverted).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESEARCH="$REPO_ROOT/ez80research"
PY="$REPO_ROOT/venv/bin/python"
RESULTS="$RESEARCH/results.tsv"
BASELINE="$RESEARCH/baseline.json"
RUNLOG="$RESEARCH/run.log"
EVALLOG="$RESEARCH/eval.log"

LABEL="${1:-unlabeled}"
EPOCHS="${EPOCHS:-120}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-600}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8000}"
MARGIN="${MARGIN:-0.005}"
DATA="${DATA:-training_data.txt}"

# --- grader tripwire: refuse to run if the agent edited the grading harness ---
GRADER_RE='^(sizes\.py|intkernel\.py|faithgate\.py|ez80interp\.py|ez80research/(evaluate|budget)\.py|ez80research/run_experiment\.sh|ez80research/program\.md)$'
if git status --porcelain | awk '{print $2}' | grep -Eq "$GRADER_RE"; then
    echo "ABORT: grader file modified — refusing to run experiment." >&2
    git status --porcelain | grep -E "$GRADER_RE" >&2
    exit 2
fi

# --- 1. capture pre-state ---
GIT_BEFORE="$(git rev-parse HEAD)"
FILES_CHANGED="$(git diff --name-only HEAD | paste -sd, - )"
[ -z "$FILES_CHANGED" ] && FILES_CHANGED="-"

# Safety net: if the run is interrupted (Ctrl-C / kill) AFTER we commit the edit
# (step 2) but BEFORE the keep/revert decision (step 7), the experiment commit
# would otherwise linger as HEAD and silently become the next baseline. The trap
# reverts to the pre-experiment commit unless the experiment was explicitly kept.
KEPT=0
trap '[ "${KEPT}" = "1" ] || { echo "[interrupt] reverting to ${GIT_BEFORE}" >&2; git reset -q --hard "${GIT_BEFORE}" 2>/dev/null; rm -f model.npz bin/NEO*.8xv bin/NEOCHAT.8xp 2>/dev/null; }; exit 130' INT TERM

# --- 2. commit the edit first (so every experiment is one revertible commit) ---
git add -A
if git diff --cached --quiet; then
    echo "(no source changes to commit — running anyway, e.g. baseline)"
else
    git commit -q -m "exp: $LABEL"
fi

# --- 3. bounded training (always from scratch) ---
# train.py auto-resumes from neochat_model.pt whenever it exists, so a leftover
# checkpoint from the previous experiment would be fine-tuned instead of the
# edited code being trained fresh — path-dependent, non-comparable results.
# Delete all prior model artifacts so every experiment is a clean EPOCHS run.
rm -f neochat_model.pt model.npz bin/NEO*.8xv bin/NEOCHAT.8xp
echo "[train] epochs=$EPOCHS timeout=${TRAIN_TIMEOUT}s data=$DATA (from scratch)"
TRAIN_START=$SECONDS
# Send SIGINT (not the default SIGTERM) on timeout: train.py catches KeyboardInterrupt,
# breaks the epoch loop, and still saves a (best/latest) checkpoint, so an experiment
# that hits the wall-clock ceiling is evaluated on what it trained instead of vanishing.
timeout -s INT "${TRAIN_TIMEOUT}"s "$PY" train.py -f "$DATA" \
    --epochs "$EPOCHS" --save-best --quant-target "$EPOCHS" \
    > "$RUNLOG" 2>&1
TRAIN_RC=$?
TRAIN_SECS=$(( SECONDS - TRAIN_START ))
EPOCHS_RUN="$(grep -Eo 'Epoch [0-9]+' "$RUNLOG" | tail -1 | grep -Eo '[0-9]+' || echo 0)"
# 124 = timeout fired; 130 = process exited on the SIGINT we sent (graceful save).
{ [ "$TRAIN_RC" -eq 124 ] || [ "$TRAIN_RC" -eq 130 ]; } && \
    echo "[train] hit wall-clock timeout — saved partial checkpoint and evaluating it"

# --- 4. evaluate (feasibility + quality) ---
EVAL_SAMPLES="$EVAL_SAMPLES" "$PY" "$RESEARCH/evaluate.py" \
    --model neochat_model.pt --npz model.npz --bin-dir bin \
    --samples "$EVAL_SAMPLES" > "$EVALLOG" 2>&1
EVAL_RC=$?

# --- 5. parse the single VERDICT line ---
VERDICT="$(grep '^VERDICT ' "$EVALLOG" | tail -1)"
echo "$VERDICT"
get() { echo "$VERDICT" | grep -oE "$1=[^ ]+" | head -1 | cut -d= -f2-; }
PASS="$(get pass)";        PASS="${PASS:-0}"
INTACC="$(get intacc)";    INTACC="${INTACC:-0.0000}"
RAM_KB="$(get ram_kb)";    RAM_KB="${RAM_KB:-0.0}"
N_APPVARS="$(get appvars)"; N_APPVARS="${N_APPVARS:-0}"
MAXAV_KB="$(get maxav_kb)"; MAXAV_KB="${MAXAV_KB:-0.0}"
REASONS="$(get reasons)";  REASONS="${REASONS:--}"
REPORTED_ACC="$(grep -oE 'reported [0-9.]+' "$EVALLOG" | tail -1 | awk '{print $2}')"
REPORTED_ACC="${REPORTED_ACC:-0.0000}"

# --- 6. current best IntAcc among kept rows (else baseline) ---
BEST=0.0
if [ -f "$RESULTS" ]; then
    BEST="$(awk -F'\t' 'NR>1 && $14==1 {if($8>m)m=$8} END{printf "%.4f", m+0}' "$RESULTS")"
fi
if [ -f "$BASELINE" ]; then
    B="$(grep -oE '"intacc"[ ]*:[ ]*[0-9.]+' "$BASELINE" | grep -oE '[0-9.]+')"
    awk "BEGIN{exit !($B>$BEST)}" && BEST="$B"
fi

# --- 7. keep / revert decision ---
KEPT=0
DECISION="revert"
if [ "$PASS" = "1" ] && awk "BEGIN{exit !($INTACC > $BEST + $MARGIN)}"; then
    KEPT=1
    DECISION="keep"
else
    # revert to pre-experiment state and clear the rejected build artifacts
    git reset -q --hard "$GIT_BEFORE"
    rm -f model.npz bin/NEO*.8xv bin/NEOCHAT.8xp 2>/dev/null
fi
EXP_ID="$(git rev-parse --short HEAD)"

# --- 8. append one row (header on first write) ---
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
    flock 9
    if [ ! -f "$RESULTS" ]; then
        printf 'ts\texp_id\tlabel\tfiles_changed\tepochs_run\ttrain_secs\tpass\tintacc\treported_acc\tram_kb\tn_appvars\tmaxav_kb\treasons\tkept\tprev_best\n' >> "$RESULTS"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$TS" "$EXP_ID" "$LABEL" "$FILES_CHANGED" "$EPOCHS_RUN" "$TRAIN_SECS" \
        "$PASS" "$INTACC" "$REPORTED_ACC" "$RAM_KB" "$N_APPVARS" "$MAXAV_KB" \
        "$REASONS" "$KEPT" "$BEST" >> "$RESULTS"
} 9>>"$RESULTS.lock"

echo "[decision] $DECISION  intacc=$INTACC best=$BEST margin=$MARGIN kept=$KEPT reasons=$REASONS"
exit 0
