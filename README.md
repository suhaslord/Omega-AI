# Omega AI

An early Python experiment in character-level conversational modeling with TensorFlow/Keras and NumPy.

## Current status

**Incomplete learning prototype.** The repository contains a single [main.py](main.py) script and five example conversation pairs. It is not a working general-purpose chatbot.

The current implementation has unresolved data-shape issues: variable-length character sequences are converted directly into NumPy arrays, while the dense model expects a fixed-width input. Generation also returns only one character and does not handle unseen characters. These issues must be resolved before the training and chat loop can run reliably.

## Intended workflow

1. Build a character vocabulary from the example pairs.
2. Convert prompts and responses to numeric sequences.
3. Train a small dense neural network.
4. Read terminal input and predict a response; `exit` leaves the intended chat loop.

## Dependencies and next steps

The script imports `numpy` and `tensorflow`; dependency versions are not pinned. A reproducible environment, compatible sequence representation, and corrected training targets are needed before a supported setup command can be provided.

For a later AI evaluation project, see [AbstainBench](https://github.com/suhaslord/AbstainBench).
