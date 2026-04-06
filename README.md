# eZ80-ai: A tiny conversational AI that targets the TI-84 Plus CE
A language model on a graphing calculator? More likely than you think!


### Disclaimer
This project was built on the shoulders of giants. Two other projects served as the foundation for this project:
- [HarryR/z80ai](https://github.com/HarryR/z80ai): The core of this hard fork, and the source of the black magic that makes this possible. Much of the code here is the same, just adapted for the eZ80 and TI-OS.
- [marspa73/atarijam](https://github.com/marspa73/atarijam) JAM (Just Atari Model), a similar project targeting late-70s Atari Microcomputers. eZ80-ai's attention mechanism comes from here.

If you like this project, please show some appreciation for those two as well. They deserve it.


### Get Started
Binaries can be found in the releases. Inside the archive you'll find:
- **NEOCHAT.8XP**: The engine. This is what you run to actually start chatting.
- **NEOA-D.8XV**: These are the model weights. They're too large to be stored within the executable, so they're APPVARs instead.

I highly recommend that you have ample flash (a few MB) and RAM (at least 150kb) available.

If your calculator is on OS 5.5 or later, you'll need to jailbreak it before running this program.
> **Important:** If you archive the APPVARs, they'll be unarchived when you start the program.
> When you (gracefully) close it, re-archiving will be attempted. If your flash is low, you may get a prompt asking for "Garbage Collection". **Always say yes**, this is just TI-OS wanting to defragment your flash before archiving the weights again.


### What to Expect
A charming little chatbot that gives surprisingly coherent (but factually incorrect) responses.

### What not to Expect
A LLM on your calculator. Sorry, this isn't going to help you cheat. It's far too stupid for that.

Lightning-fast inference. Expect seconds per character, not characters per second. It's not slow enough to bore you though!


### Contributing
Contributions are welcome! Contributors should know that **most of the inference and training scripts currently target Intel Arc**, as I've been using the Arc Pro B50 for this. It shouldn't be difficult to add support for ROCm and CUDA for training, and for inference purposes there's CPU fallback as well.

CLAUDE.md actually has all the information you'd need to know as a contributor, so read that to get an idea of the pipeline here.


### Roadmap
There's only one "solid" feature that I'd like to ship eventually: TI-84 Evo support.

So far, all we have are leaks, and God knows if we'll be able to run assembly on it. Brighter minds than I will likely be working on that capability as soon as they can get their hands on one. It's also possible that they ditch the Z80 lineage and move on to a more modern chip, which could throw a wrench into my plans.

Other than that, I'll see if there's anything I can do to the model architecture to improve accuracy further within our very tiny footprint, though I already tried about a million things on that front. Like I said, contributions are welcome!


### Licensing
Both of the parent projects are MIT-licensed, and eZ80-ai is no different.
