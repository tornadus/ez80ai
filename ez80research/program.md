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
4. **Fixed I/O contract.** Output charset stays the exact 43 chars incl. EOS
   (`train.CHARSET`); input encoding stays 256-dim (128 query + 128 context,
   `INPUT_SIZE`). Do not shrink the output space or change the tokenizer to
   inflate accuracy.
5. **Grader files are OFF-LIMITS.** Never edit `sizes.py`,
   `ez80research/evaluate.py`, `ez80research/budget.py`,
   `ez80research/run_experiment.sh`, or this `program.md`. The runner aborts if
   any grader file is modified.

## Levers you MAY change (the whole search space)

Read `train.py`, `libqat.py`, and `encoding.py` to see exact names/locations.

- **Hyperparameters** — lr, batch size, the quant/overflow loss weights and the
  quantization-temperature ramp in the training loop, dual-bias threshold.
  (Keep `EPOCHS` at the loop default — it's the fairness budget; the baseline was
  measured at it.)
- **Architecture** — `HIDDEN_SIZES` near the top of `train.py`. Width changes
  that keep the 3-hidden-layer topology are the safe lever and trade IntAcc
  against the RAM budget (the core tension of this project). NOTE:
  `buildchat84.py` emits a fixed NEOA-D AppVar layout and splits layer 2 as
  `[:256]/[256:]`; changing the layer COUNT, or widths large enough that a single
  layer's weights exceed an AppVar's 65535-byte limit, requires generalizing the
  build's splitter too. The harness rejects anything that doesn't build, so
  generalizing the splitter to shard any layer across multiple AppVars is itself
  a legitimate, high-value experiment.
- **Encoding** — `encoding.py` hashing/bucketing (keep `input_size == 256`).
- **Quant / training internals** — `libqat.py` 2-bit quantization, overflow
  regularization, `ACTIVATION_SCALE`.
- Anything else, as long as the five hard constraints hold.

Because the baseline already nearly fills RAM, "smaller-but-smarter" (better
accuracy at equal-or-less size) is as valuable as anything — don't fixate on
just making the net wider.

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
- Do not special-case the evaluator's sample, hardcode outputs, weaken the
  contract, or edit grader files. Because evaluation is empirical, cheating is
  an automatic FAIL.
