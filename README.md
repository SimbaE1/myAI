# myAI

An AI… that I built!

It’s not the best or the fastest AI, nor the cheapest—but I made it, and that counts for something.

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Training](#training)  
  - [Inference](#inference)  
- [Training Details](#training-details)  
- [Inference Details](#inference-details)  
- [Contributing](#contributing)  
- [License](#license)  

## Overview
**myAI** is a token-level GPT trainer and text generator implemented in Python with PyTorch and Hugging Face Transformers. It’s designed to be easy to run, experiment with, and extend.

## Features
1. **Bad Word Detector**  
   Filters out profanities so the AI will never utter a bad word (as long as my profanity library is up to date).  
2. **Easy-Use CLI**  
   Train and generate text with simple commands—no complicated setup.  
3. **Multiple Dataset Support**  
   Pick from `tiny-shakespeare`, `wikitext-103`, `bookcorpus`, or point at any local text file.  
4. **Configurable Hyperparameters**  
   Tweak embedding size, layer count, number of heads, learning rate, etc., right in the code.

## Installation
```bash
git clone https://github.com/SimbaE1/myAI.git
cd myAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> **Optional:** Install the PyTorch build with CUDA or MPS support if you have a compatible GPU.

## Usage

### Training
```bash
# Train on Tiny-Shakespeare for 1 epoch (1000 steps)
python ai1.py --train -d tiny-shakespeare -e 1

# Train on WikiText-103 for 5 epochs
python ai1.py --train -d wikitext-103 -e 5

# Train indefinitely on BookCorpus
python ai1.py --train -d bookcorpus -e inf
```
By default, each run saves its weights to `gpt_<dataset>.pth` in the project root.

### Inference
```bash
# Complete a single prompt
python ai1.py --complete --prompt "Once upon a time"

# Launch the interactive REPL
python ai1.py --complete
```
You can pass flags for `--temperature`, `--top_k`, and `--top_p` in `ai1.py` to control sampling.

## Training Details
- **Architecture:** 6-layer Transformer, 8 attention heads, 512-dimensional embeddings  
- **Batching:** Sliding-window over the dataset for low memory usage  
- **Monitoring:** Tracks a running mean of the last 100 losses for smoother metrics

## Inference Details
- **Profanity Filter:** Zeroes out bad-word token logits before sampling  
- **Sampling:** Supports temperature, top-k, and nucleus (top-p) sampling  
- **No-Repeat:** Implements a no-repeat-ngram constraint to avoid duplicated tokens

## Contributing
Contributions welcome!  
1. Fork this repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/YourFeature
   ```  
3. Commit your changes:  
   ```bash
   git commit -m "Add awesome feature"
   ```  
4. Push to your fork:  
   ```bash
   git push origin feature/YourFeature
   ```  
5. Open a Pull Request against `main`.

Please follow the existing code style and include tests or examples where appropriate.

## Colab AI
This runs in colab. Your model should be in the file system, in which it will appear in /content/[NAME]
Passes the restrictions of colab, so you can run it on their GPU/TPU

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
