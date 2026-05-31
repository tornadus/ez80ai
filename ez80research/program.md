# NEOCHAT research brief

You are an autonomous ML research agent improving the NEOCHAT model — a
character-level language model that runs on a **Ti-84 Plus CE calculator** in
**pure 24-bit signed integer math** (no floats on device). Iterate continuously:
form one hypothesis, change the code, run one experiment, keep it if it improved,
revert it if not. Do not ask for confirmation.

## Objective

**Maximize integer-inference accuracy (`IntAcc`) of a model that actually fits
and runs on a real Ti-84 Plus CE.** IntAcc is the next-character accuracy of the
*integer* inference path (`use_int=True`), measured independently by the harness
(`ez80research/evaluate.py`). Higher is better. There is exactly one objective.

## Hard constraints — NEVER break these (any violation discards the experiment)

1. **Integer-only inference.** The metric is measured with `use_int=True`. You
   cannot "improve" by making the float path better or drifting toward floats.
2. **Must export and build end-to-end with zero errors.** The harness runs the
   real `exportmodel.py` then `buildchat84.py`. A model that doesn't build is
   not a model.
3. **Must fit the calculator's RAM.** AppVars are unarchived into RAM at
   runtime, so program code + ALL packed weights + working buffers must fit in
   ~150 KB of RAM (`sizes.RAM_BUDGET_BYTES`). This is the binding constraint and
   the one you'll fight most. (The .8xv format also caps each AppVar at 65535
   bytes, but a layer may be split across as many AppVars as needed, so that is
   a build-layout detail — RAM is what bounds feasibility.) Enforced against the
   build's real printed sizes.
4. **Fixed I/O contract.** Output charset stays the exact 43 chars incl. EOS
   (`train.CHARSET`); input encoding stays 256-dim (128 query + 128 context,
   `INPUT_SIZE`). Do not shrink the output space or change the tokenizer to
   inflate accuracy.
5. **Grader files are OFF-LIMITS.** Never edit `sizes.py`,
   `ez80research/evaluate.py`, `ez80research/budget.py`,
   `ez80research/run_experiment.sh`, or this `program.md`. The runner aborts if
   any grader file is modified.

## Levers you MAY change (the whole search space)

- **Hyperparameters** — lr, batch size, epochs (within the run budget), the
  quant/overflow loss weights (`train.py:439-440`), the quantization-temperature
  ramp (`train.py:419`), dual-bias threshold.
- **Architecture** — `HIDDEN_SIZES` (`train.py:43`). Width changes that keep the
  3-hidden-layer topology are the safe lever and trade IntAcc against the RAM
  budget (the core tension of this project). NOTE: `buildchat84.py` emits a
  fixed NEOA-D AppVar layout and splits layer 2 as `[:256]/[256:]`
  (`buildchat84.py:261-296`); changing the layer COUNT, or widths large enough
  that a single layer's weights exceed an AppVar's 65535-byte limit, requires
  generalizing the build's splitter too. The harness rejects anything that
  doesn't build, so generalizing the splitter to shard any layer across
  multiple AppVars is itself a legitimate, high-value experiment.
- **Encoding** — `encoding.py` hashing/bucketing (keep `input_size == 256`).
- **Quant / training internals** — `libqat.py` binarization, BN folding,
  overflow regularization, `ACTIVATION_SCALE`.
- Anything else, as long as the five hard constraints hold.

## Experiment protocol (one experiment = one git commit)

1. Read this file, the current `train.py` (and any file you'll edit), and the
   last ~20 rows of `ez80research/results.tsv`. Don't repeat a tried-and-failed
   idea.
2. Form **one** hypothesis. Edit the relevant production file(s).
3. Run: `bash ez80research/run_experiment.sh "<short hypothesis label>"`
4. Read the printed `VERDICT` line. The script has **already** committed the
   edit and **already** decided keep vs. revert (greedy: keep iff `pass=1` and
   `intacc` beat the current best by the margin). You do not commit or revert
   manually.
5. Go to step 1. Continue until told to stop.

## Honesty rules

- The harness re-computes IntAcc itself via the integer path and cross-checks it
  against the number `train.py` logged; a mismatch fails the experiment.
- The build is run for real; crashes and over-budget builds fail.
- Do not special-case the evaluator's sample, hardcode outputs, weaken the
  contract, or edit grader files. Because evaluation is empirical, cheating is
  an automatic FAIL.
