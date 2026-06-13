# eZ80-ai: A tiny conversational AI that targets the TI-84 Plus CE
A language model on a graphing calculator? More likely than you think!


### Disclaimer
This project was built on the shoulders of giants. Two other projects served as the foundation for this project:
- [HarryR/z80ai](https://github.com/HarryR/z80ai): The core of this hard fork, and the source of the black magic that makes this possible. Much of the code here is the same, just adapted for the eZ80 and TI-OS.
- [marspa73/atarijam](https://github.com/marspa73/atarijam) JAM (Just Atari Model), a similar project targeting late-70s Atari Microcomputers. eZ80-ai's context-attention mechanism was originally inspired by JAM (it has since been removed to keep on-device output faithful to the trained model, but JAM remains a foundational inspiration).

If you like this project, please show some appreciation for those two as well. They deserve it.


### Get Started
Binaries can be found in the releases. Inside the archive you'll find:
- **NEOCHAT.8XP**: The engine. This is what you run to actually start chatting.
- **NEO\*.8XV** (a set of AppVars named NEOA, NEOB, …): These are the model weights. They're too large to be stored within the executable, so they're APPVARs instead — the exact count depends on the model size. Transfer all of them. They live in **archived flash** and are read in place — they never take up your RAM.

I highly recommend that you have ample flash (~1.5 MB) available. RAM needs are tiny (the program itself is a few KB).

If your calculator is on OS 5.5 or later, you'll need to jailbreak it before running this program.
> **Important:** The weight APPVARs are meant to stay archived; the transfer should put them in the archive automatically. If one ends up in RAM, the program archives it on startup — if your flash is fragmented you may get a prompt asking for "Garbage Collection". **Always say yes**, this is just TI-OS defragmenting your flash before archiving the weights.


### Building from source
A fresh clone is **source only** — the trained weights (`*.pt`, `*.npz`), the
generated `training_data.txt`, and the `bin/` artifacts are git-ignored, so you
build them yourself:

```bash
pip install -r requirements.txt
python3 prepare_data.py                 # downloads nq_open, writes training_data.txt
python3 train.py -f training_data.txt --epochs 900 --save-best --quant-target 300
                                         # -> neochat_model.pt (~50 min on an Arc Pro B50)
python3 exportmodel.py                   # neochat_model.pt -> model.npz
python3 buildchat84.py --model model.npz # -> bin/NEOCHAT.8xp + NEO*.8xv weight AppVars
python3 test_model.py                    # sanity checks on the trained model
python3 test_faithfulness.py             # asserts the eZ80 build mirrors the Python sim
```

`buildchat84.py` and `exportmodel.py` only need NumPy; training/chat/eval need
PyTorch (see `requirements.txt` for the Intel Arc / XPU note).

The training schedule is 900 epochs with the quantization ramp ending at epoch
300 (`--quant-target 300`) — the remaining 600 epochs fine-tune at full
quantization, which measured ~+0.04 IntAcc over stopping at the ramp. Training
is unseeded and run-to-run spread is real (~0.05 IntAcc): train a few
candidates and keep the best one.

### What to Expect
A charming little chatbot that gives surprisingly coherent (but factually incorrect) responses.

### What not to Expect
A LLM on your calculator. Sorry, this isn't going to help you cheat. It's far too stupid for that.

~~Lightning-fast inference. Expect seconds per character, not characters per second. It's not slow enough to bore you though!~~
Update: It's pretty fast now!


### Contributing
Contributions are welcome! Contributors should know that **most of the inference and training scripts currently target Intel Arc**, as I've been using the Arc Pro B50 for this. It shouldn't be difficult to add support for ROCm and CUDA for training, and for inference purposes there's CPU fallback as well.

CLAUDE.md actually has all the information you'd need to know as a contributor, so read that to get an idea of the pipeline here.


### Roadmap
Eventually a Ti-84 Evo port would be cool, but there's no way to execute native code as of now. It would also require a complete reimplementation of pretty much everything. It makes more sense as a separate project.
Aside from that, I'd like to improve the model as much as I feasibly can.


### Licensing
Both of the parent projects are MIT-licensed, and eZ80-ai is no different.
