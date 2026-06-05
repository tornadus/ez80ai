# NEOCHAT research brief

You are an autonomous ML research agent improving the NEOCHAT model — a
character-level language model that runs on a **Ti-84 Plus CE calculator** in
**pure 24-bit signed integer math** (no floats on device). You run a continuous,
self-driving loop: form a hypothesis, run an experiment, keep it if it improved,
move on if not. You never ask for confirmation and you never stop on your own
(see "When to stop").

## Objective

**Maximize integer-inference accuracy (`IntAcc`) of a model that actually fits
and runs on a real Ti-84 Plus CE.** IntAcc is the next-character accuracy of the
*integer* inference path (`use_int=True`), measured independently by the harness
(`ez80research/evaluate.py`). Higher is better. There is exactly one objective.

The baseline (released 2-bit `[512,512,256]` arch) is frozen in
`ez80research/baseline.json` and is the first row of `ez80research/results.tsv`.
Every experiment is compared against the best `kept` IntAcc so far.

## Findings from prior sessions — READ THIS FIRST

A previous agent ran ~38 experiments + several research probes and reached a
comprehensively-validated ceiling. Start from the current best; do NOT re-run the
dead-ends below.

**Current best: IntAcc ≈ 0.6039** (baseline 0.5521, **+5.2pp / +9.4% relative**),
committed. Kept changes (all in `train.py`/`libqat.py`, compute-neutral):
- Cosine LR schedule (`CosineAnnealingLR`, T_max=epochs, **eta_min=lr*0.02**).
- quant-loss weight 0.10 → **0.25** (knee; 0.40 regresses).
- weight-quantile 0.95 → 0.90 → **0.85** (per-layer scale = Nth-pctile of |W|,
  changed in ALL 6 spots: libqat `quantize_weights_2bit` + `quantization_friendly_loss`,
  train `_forward_int`×2 + `get_quantized_params`×2). Biggest single win. Optimum is
  a flat plateau ~0.83–0.85; lower gives nothing.
- QT ramp reaches full-quant EARLY: factor 0.8 → **0.4** (QT=1.0 by epoch 0.4·epochs).
  "Quantize early and long" helps; 0.3 is too early (starves float learning).

**Winning meta-strategy:** STACK two same-direction sub-margin positives in ONE
experiment — they proved super-additive (both big jumps came from 2-change
quantization/schedule stacks). The 0.005 margin guards run-noise (train.py
`--save-best` uses an unseeded 50k eval, so single tweaks read ±0.002–0.005).

**THE CEILING — why almost everything reverts.** Integer accuracy is capped ~0.60
= architecture/data **float ceiling ~0.63** minus a **structural ~3pp
train(float)/eval(integer) gap**. Float gains DO NOT transfer to the integer path:
4-gram encoding (float 0.634, int flat), 3-bit weights (float 0.639, int +0.003),
per-channel scales (float 0.646, int 0.53–0.57), 4-layer depth (worse on both). The
gap exists because training uses the float forward (effective weights w_quant·scale
≈0.05) but the metric is the integer path (unscaled integer weights, ×32 input,
÷4/layer). The only gap-closer — integer-aware training — FAILS: standard STE blows
up (gradient ~1/scale ≈20× too large), and gradient-scale-corrected (k=1) STE
collapses to ~0.107. Bias-only int fine-tune also hurts (0.588).

**DEAD-ENDS — already tested, do NOT repeat:**
- *Capacity:* width (L3 256→300 neutral), depth (4-layer [384,384,384,256] worse on
  float AND int), 3-bit weights (+0.003). 2-bit capacity is NOT the bottleneck —
  extra params just add quantization noise.
- *Per-channel / learned scales:* per-output-channel scale + per-neuron power-of-2
  shift (0.53–0.57), LSQ learned scales (0.553). eZ80 can only shift (power-of-2);
  exact per-channel scales aren't realizable and the gain evaporates.
- *Activations:* ACTIVATION_SCALE 32→64 (neutral; faithful change = train.py const +
  `buildchat84.py` ~line 975 bucket-increment constant), ÷4→÷2 (neutral),
  activation-aware QAT (0.596), output-÷4 removal (0.594).
- *Schedule:* linear LR (0.5985), 5-epoch warmup (0.6012), LR peak 0.003 (0.591 —
  model is sensitive to high early LR), QT start 0.3→0.5 (0.597); eta_min already
  optimal at 0.02.
- *Optimization:* Adam beta2=0.99 (0.594), grad-clip 1.0 (0.597), **focal loss
  (0.501** — hard examples are unlearnable conflicting labels), **stochastic rounding
  (0.571** — noise hurts the in-sample fit), weight-init ×1.5 (0.597), fixed
  selection-eval subset (0.595), late-ramped quant-loss (0.600).
- *weight_decay is LOAD-BEARING for quantization* (keeps weights compact for clean
  rounding): wd=0 → 0.444 catastrophe, wd=2e-4 → 0.575. Keep 1e-4.
- *Encoding is not the bottleneck* (inputs are distinct; and the eval encoder params
  num_buckets=128/128 & context_len=8 are FIXED inside evaluate.py — unchangeable).
- *dual_bias_threshold* 3→5 (0.595; build hardcodes `cp_n(3)` at buildchat84.py:603).

**Key gotchas:**
- The metric is effectively **IN-SAMPLE** (evaluate.py samples the same
  training_data.txt). It rewards fitting and PUNISHES regularization/noise (dropout,
  label smoothing, stochastic rounding, focal all hurt).
- **Wall-clock ~600s (TRAIN_TIMEOUT) is a second binding constraint:** batch 4096 /
  slower nets time out → NO_CHECKPOINT. Keep experiments compute-neutral. Raising
  TRAIN_TIMEOUT for a slow config can end a subagent's turn mid-run, leaving a
  DANGLING committed edit.
- IntAcc is measured ONLY by `train.py._forward_int` (the build is a feasibility/size
  gate; its numerics aren't compared). Any `_forward_int` change MUST be mirrored
  faithfully in `buildchat84.py` codegen, or it's cheating.
- Use **research probes** (edit → train.py directly → read IntAcc → revert, WITHOUT
  the harness) to size build-coupled ideas (encoding, bit-width, ÷-changes,
  per-channel) before investing in build surgery.

**To beat 0.6039 you almost certainly need a NEW DEGREE OF FREEDOM** (a relaxed hard
constraint): >2-bit weights on the big layers (needs more RAM), a fundamentally
different integer-aware-training scheme that actually closes the gap, cleaner/larger
training data, or build generalization enabling architectures the fixed NEOA-D layout
forbids. Within the current hard constraints, the search space is exhausted — but per
"When to stop", keep generating genuinely-novel ideas anyway.

**Sim vs device (learned the hard way):** IntAcc is the *simulation* (`_forward_int`).
The harness builds the binary but does NOT compare its numerics to the sim, so a
device/sim divergence is SILENT. A released build once ran fine in the sim yet emitted
on-calc gibberish, because `buildchat84.py` carried device-only mechanisms (context
attention, EOS/confidence/repeat heuristics) absent from the sim, plus an eZ80
16-bit-counter quirk corrupting a weight pointer. The fix was to make the calc a faithful
mirror of the sim (pure argmax + dual bias, full 24-bit counters). If you change
`_forward_int`, mirror it in `buildchat84.py`; never add device-only behavior.

## Hard constraints — NEVER break these (any violation discards the experiment)

1. **Integer-only inference.** The metric is measured with `use_int=True`. You
   cannot "improve" by making the float path better or drifting toward floats.
2. **Must export and build end-to-end with zero errors.** The harness runs the
   real `exportmodel.py` then `buildchat84.py`. A model that doesn't build is
   not a model.
3. **Must fit the calculator's RAM.** AppVars are unarchived into RAM at
   runtime, so program code + ALL packed weights + working buffers must fit in
   ~150 KB of RAM (`sizes.RAM_BUDGET_BYTES`). This is the binding constraint and
   the one you'll fight most — the baseline already uses ~143 KB, so there's
   little headroom. (The .8xv format also caps each AppVar at 65535 bytes, but a
   layer may be split across as many AppVars as needed, so that is a build-layout
   detail — RAM is what bounds feasibility.) Enforced against the build's real
   printed sizes.
4. **Fixed output contract (the TASK).** Output charset stays the exact 43 chars
   incl. EOS (`train.CHARSET`); do NOT shrink the output space. Input ENCODING is
   now a degree of freedom (see Levers), but you cannot game the metric with it:
   the harness rebuilds its eval encoder from your spec and the teacher-forced
   example construction keeps the next-char label out of the input.
5. **Grader files are OFF-LIMITS.** Never edit `sizes.py`, `intkernel.py`,
   `faithgate.py`, `ez80interp.py`, `ez80research/evaluate.py`,
   `ez80research/budget.py`, `ez80research/run_experiment.sh`, or this
   `program.md`. The runner aborts if any is modified. (`modelspec.py` is YOURS —
   that is where you run experiments.)

## Levers you MAY change (the whole search space)

**Everything about the model now lives in ONE file: `modelspec.py`.** An
experiment = edit `modelspec.DEFAULT_SPEC`. The resolved spec is baked into the
checkpoint/`.npz` and read by EVERY stage (train, eval, export, build, the
faithfulness gate), so a knob set once propagates everywhere — no more editing the
same constant in 6 places or hand-mirroring the build. (Only a genuinely NEW
mechanism — a new activation, a new quant scheme — needs code, and it must go in
BOTH `intkernel.py` and the codegen; but those are grader-owned, so prefer spec
changes.)

Spec knobs (see `modelspec.DEFAULT_SPEC` for names + defaults):
- **Architecture** — `hidden_sizes` (ANY depth/width; the build auto-shards any
  layer across AppVars), `activation`.
- **Quantization** — per-layer `weight_bits` (2/3/4-bit, mixed precision OK),
  `weight_quantile`, per-layer `inter_layer_shift`, `activation_scale`.
- **Output bias** — `dual_bias_threshold`.
- **Encoding** (now a real DOF) — `query_buckets`, `context_buckets` (powers of
  two, ≤ 256), `context_len`. Query stays trigram, context stays 1..N-gram.
- **Training** — `lr`, `batch_size`, `weight_decay`, `quant_loss_weight`,
  `qt_start`, `qt_ramp_factor`, `eta_min_frac`, `epochs`.
- **Compute budget** — `compute_budget = {mode, limit}`. `grad_steps` gives every
  arch the SAME number of optimizer steps (fair across slow/fast nets — a slower
  but better model is no longer silently penalized); `wall_s` is the legacy cap.

**The faithfulness gate (the big change).** The harness now EXECUTES the real
emitted eZ80 machine code (`faithgate`/`ez80interp`) and requires it to match the
integer reference (`intkernel.forward_device`) EXACTLY; a `FAITH_FAIL` discards
the experiment. Build-coupled changes are therefore no longer silent — if your
codegen disagrees with the sim you find out immediately, so the on-calc-gibberish
class of bug is gone. A spec-only RAM PRE-GATE also rejects infeasible models in
<1s before training (look for `[pregate]`).

**Dead-ends now genuinely RE-testable** (they were measured on the trunc sim, or
were unbuildable; faithfulness now guarantees device truth): depth / layer count,
>2-bit & mixed-precision weights, encoding geometry, per-layer shift /
`activation_scale`. NOTE: 4-bit DOUBLES a layer's bytes, so big layers won't fit —
mixed precision (4-bit only on small/output layers) is the feasible play, and is
`program.md`'s nominated path past the ~0.60 ceiling.

Because the baseline already nearly fills RAM, "smaller-but-smarter" (better
accuracy at equal-or-less size) is as valuable as anything.

## How you operate — the loop

You are the **strategist**. You hold the only long-running thread: the objective,
the history of what's been tried (from `results.tsv`), and the next idea. You do
**not** do the gruntwork yourself. **Run each experiment in a fresh subagent** so
your own context stays clean and strategic while the heavy, throwaway work
happens elsewhere.

Each iteration:

1. **Think (you, main context).** Look at `ez80research/results.tsv` — each row
   is `label, intacc, kept, reasons, …` — and the current best. Form **one**
   concrete hypothesis for raising IntAcc within the constraints. Don't repeat an
   idea that already failed; build on what worked.
2. **Delegate (spawn a subagent).** Launch a subagent with a self-contained task.
   Give it: the hypothesis, what to change (and where, if you know), and the
   instruction to run the experiment. The subagent, in its own fresh context:
   reads the relevant code, makes the edit, runs
   `bash ez80research/run_experiment.sh "<short hypothesis label>"`, reads the
   printed `VERDICT` line, and returns to you **only**: (a) the VERDICT line,
   (b) the keep/revert decision the script printed, and (c) one sentence on what
   it changed. It must not iterate, ask questions, or do anything beyond that one
   experiment.
3. **Absorb (you).** Update your mental model from the returned verdict. The
   script has **already** committed the edit and decided keep-vs-revert (greedy:
   keep iff `pass=1` and `intacc` beat the best by `MARGIN`) and logged the row —
   you never touch git yourself. Form the next hypothesis.
4. **Repeat — indefinitely.**

Why subagents: the code-reading, file edits, training logs, build output, and
verdict parsing are bulky, single-use context. Keeping them out of your thread
lets you reason over the entire experiment history without drowning in
transcripts. `results.tsv` + git history are your durable memory; each
subagent's context is disposable and discarded after it reports back.

## When to stop

**Never — except for an infrastructure failure that genuinely prevents
progress.** Keep iterating indefinitely. The ONLY legitimate stop condition is an
OS- or hardware-level failure: the disk fills, the machine runs out of memory and
the OS kills training, the Python interpreter / GPU / venv disappears, the repo
becomes unreadable — something that makes running *any* experiment impossible.

A failed **experiment** is NORMAL and is never a reason to stop. `pass=0` from
being over RAM budget, a build crash, a contract violation, or simply not beating
the baseline are all expected, useful outcomes — they get logged and you move to
the next idea. Do not stop because results plateau, because many experiments fail
in a row, because you're low on ideas, or because you feel "done." There is no
done. If you're out of obvious ideas, generate less obvious ones (new encodings,
quantization tricks, the AppVar-splitter generalization, regularization, LR
schedules, architectural reshaping within budget).

## Honesty rules

- The harness re-computes IntAcc itself via the integer path and cross-checks it
  against the number `train.py` logged; a mismatch fails the experiment.
- The build is run for real; crashes and over-budget builds fail.
- The faithfulness gate executes the REAL emitted machine code and requires it to
  match the integer reference exactly (`FAITH_FAIL` otherwise) — you cannot ship a
  build whose numerics diverge from the sim, even silently.
- Do not special-case the evaluator's sample, hardcode outputs, weaken the
  contract, or edit grader files. Because evaluation is empirical, cheating is
  an automatic FAIL.
