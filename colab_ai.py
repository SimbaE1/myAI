# -----------------------------------------------
# Configuration (set these before running in Colab)
MODE = "train"       # "train" or "complete"
DATASET = "/content/bookcorpus"  # "tiny-shakespeare", "wikitext-103", "bookcorpus", or path to local .txt
EPOCHS = "inf"            # Number of epochs (int) or "inf"
PROMPT = "Once upon a time, "           # Prompt text to complete in "complete" mode
CPU_PERCENT = 100     # Percent of CPU threads to use
GPU_PERCENT = 100     # Percent of GPU memory to use (applies to CUDA/MPS)
# -----------------------------------------------
# Suppress all warnings (including urllib3’s NotOpenSSLWarning) so our prompt isn’t drowned out
import warnings
warnings.filterwarnings("ignore")

# In Colab, set CPU/GPU percentages via environment vars
import os
cpu_percent = int(os.getenv("CPU_PERCENT", str(CPU_PERCENT)))
gpu_percent = None
if "GPU_PERCENT" in os.environ:
    try:
        gpu_percent = int(os.getenv("GPU_PERCENT"))
        gpu_percent = max(1, min(100, gpu_percent))
    except ValueError:
        gpu_percent = GPU_PERCENT
else:
    gpu_percent = GPU_PERCENT
#!/usr/bin/env python3
"""
ai1.py – Minimal character‑level GPT trainer & completer
-------------------------------------------------------
Usage:
  python ai1.py train -e 5                 # 5 epochs on default tiny‑shakespeare
  python ai1.py train -d wikitext-103 -e inf   # infinite epochs on WikiText‑103
  python ai1.py complete --prompt "Hello world"
                                      # loads saved weights and continues text
"""

from transformers import (
    AutoTokenizer,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
import os
from typing import Optional
import torch

# Configure CPU threading before any PyTorch work
total_cores = os.cpu_count() or 1
threads_to_use = max(1, int(total_cores * cpu_percent / 100))
torch.set_num_threads(threads_to_use)
try:
    torch.set_num_interop_threads(threads_to_use)
except RuntimeError:
    pass  # inter-op threads must be set before any parallel work
print(f"Using {threads_to_use} of {total_cores} CPU threads ({cpu_percent}%)")
from datasets import load_dataset, disable_caching
from datasets import DownloadConfig
import glob
from better_profanity import profanity
from tqdm import tqdm
import sys


# -------------------------- Hyper‑parameters -------------------------- #
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ------------------------ Profanity filter setup ------------------------ #
# We keep the profanity word list *out* of our own source by pulling it
# from the maintained “better_profanity” library.
#   pip install better_profanity
profanity.load_censor_words()                 # loads its default English list
BAD_WORDS = profanity.CENSOR_WORDSET          # this is a `set[str]`

 # Convert every token in the GPT‑2 vocabulary into text and block it if that
# text *contains* any forbidden word as a substring (case‑insensitive).  This
# captures tokens like "shit", "shitty", "bullshit", "cocky", etc., even when
# the bad word is only part of the token.
BAD_WORD_IDS: set[int] = set()
# Ensure each item is converted to a plain string before lower‑casing
lower_bad_words = [str(w).lower() for w in BAD_WORDS]

for tok_id in range(tokenizer.vocab_size):
    tok_text = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False).lower()
    tok_text = tok_text.lstrip()          # strip leading whitespace BPE often adds
    for bad in lower_bad_words:
        if bad in tok_text:
            BAD_WORD_IDS.add(tok_id)
            break  # no need to check other bad words for this token

BAD_WORD_IDS = sorted(BAD_WORD_IDS)
VOCAB_SIZE = tokenizer.vocab_size
BLOCK_SIZE = 1024
BATCH_SIZE = 4
EMBED_DIM = 512
N_LAYERS = 6
N_HEADS = 8
LR = 3e-4

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

MODEL_PATH = "char_gpt.pth"
MPS_EMPTY_CACHE = True
disable_caching()

# --------------------------- Model definition ------------------------ #
class GPTConfig:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.block_size = BLOCK_SIZE
        self.n_layer = N_LAYERS
        self.n_head = N_HEADS
        self.n_embd = EMBED_DIM


class GPT(torch.nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = torch.nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.drop = torch.nn.Dropout(0.1)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=cfg.n_embd,
                    nhead=cfg.n_head,
                    dim_feedforward=4 * cfg.n_embd,
                    batch_first=True,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = torch.nn.LayerNorm(cfg.n_embd)
        self.head = torch.nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        x = self.embed(idx) + self.pos[:, :t, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


# --------------------------- Data pipeline --------------------------- #
def encode_to_tensor(text: str) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)


def stream_batches(dataset_choice: str):
    if dataset_choice == "tiny-shakespeare":
        print("⏳ Loading Tiny‑Shakespeare corpus …")
        try:
            ds = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True, ignore_verifications=True)
            text = ds[0]["text"]
        except Exception as e:
            print("⚠️  Online download failed:", e)
            print("➡️  Falling back to bundled tiny‑shakespeare …")
            text = (
                "From fairest creatures we desire increase,\n"
                "That thereby beauty's rose might never die,\n"
                "But as the riper should by time decease,\n"
                "His tender heir might bear his memory …\n"
            ) * 200
    elif dataset_choice == "bookcorpus":
        print("⏳ Streaming BookCorpus (≈1 GB) …")
        try:
            download_config = DownloadConfig(
                cache_dir="/tmp/hf_stream_cache",  # ephemeral Colab storage
                force_download=True                # bypass any local cache
            )
            ds_iter = load_dataset(
                "lucadiliello/bookcorpusopen",     # original mirror if desired
                split="train",
                streaming=True,
                download_config=download_config
            )
            # ds_iter = load_dataset(
            #   "rojagtap/bookcorpus",      # alternate mirror
            #   split="train",
            #   streaming=True
            # )
        except Exception as e:
            raise RuntimeError(
                "Failed to load BookCorpus mirror; please check your Internet "
                "connection or choose a different dataset."
            ) from e

        # Build a small rolling buffer of token ids
        buf: list[int] = []
        needed = BLOCK_SIZE * BATCH_SIZE
        for ex in ds_iter:
            line = ex["text"]
            buf.extend(tokenizer.encode(line, add_special_tokens=False))
            while len(buf) >= needed:
                chunk, buf = buf[:needed], buf[needed:]
                x = torch.tensor(chunk, dtype=torch.long).view(BATCH_SIZE, BLOCK_SIZE)
                y = x.clone()
                yield x, y
    elif dataset_choice == "wikitext-103":
        print("⏳ Loading WikiText‑103 (≈500 MB) …")
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text = "\n".join(ds["text"])
    else:
        # Stream .txt files in directory or single file without loading all into memory
        buf: list[int] = []
        needed = BLOCK_SIZE * BATCH_SIZE

        def file_line_generator(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    yield line

        # Determine list of files
        if os.path.isdir(dataset_choice):
            print(f"⏳ Streaming all text files in directory: {dataset_choice} …")
            file_list = sorted(glob.glob(os.path.join(dataset_choice, "*.txt")))
        else:
            file_list = [dataset_choice]

        # Stream through files and yield batches
        while True:
            for path in file_list:
                print(f"Loading {path}…")
                for line in file_line_generator(path):
                    buf.extend(tokenizer.encode(line, add_special_tokens=False))
                    while len(buf) >= needed:
                        chunk, buf[:] = buf[:needed], buf[needed:]
                        x = torch.tensor(chunk, dtype=torch.long).view(BATCH_SIZE, BLOCK_SIZE)
                        y = x.clone()
                        yield x, y
            # After all files, continue streaming by looping over files again


# --------------------------- Training loop --------------------------- #
def train_one_epoch(model: GPT, dataset_choice: str):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_per_epoch = 1000

    data_iter = stream_batches(dataset_choice)
    print("⏳ Warming up first batch…")
    first_x, first_y = next(data_iter)
    print("✅ First batch ready — starting training")
    recent_losses: list[float] = []
    WINDOW = 100  # how many recent batches to average

    pbar = tqdm(range(steps_per_epoch), desc="training", leave=True)
    for step in pbar:
        x, y = (first_x, first_y) if step == 0 else next(data_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Track running window of the last WINDOW raw loss values
        recent_losses.append(loss.item())
        if len(recent_losses) > WINDOW:
            recent_losses.pop(0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if MPS_EMPTY_CACHE and DEVICE == "mps":
            torch.mps.empty_cache()

        pbar.write(f"batch {step + 1} completed")
        if len(recent_losses) == WINDOW:
            mean_loss = sum(recent_losses) / WINDOW
            pbar.set_postfix(raw_loss=f"{loss.item():.10f}",
                             mean_loss=f"{mean_loss:.10f}")
        else:
            # not enough batches yet to fill window
            pbar.set_postfix(raw_loss=f"{loss.item():.4f}",
                             mean_loss="n/a")

    torch.save({"state_dict": model.state_dict()}, MODEL_PATH)
    print(f"\n✅ Saved weights to {MODEL_PATH}")


# --------------------------- Generation ------------------------------ #
@torch.inference_mode()
def generate_stream(
    model: GPT,
    prompt: str,
    max_new: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.25,
):
    model.eval()
    ids = encode_to_tensor(prompt).unsqueeze(0).to(DEVICE)
    processors = LogitsProcessorList(
        [
            NoRepeatNGramLogitsProcessor(ngram_size=3),
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
        ]
    )
    print(prompt, end="", flush=True)

    for _ in range(max_new):
        logits, _ = model(ids[:, -BLOCK_SIZE:], None)
        next_logits = logits[0, -1] / temperature
        # 🚫 Block all forbidden tokens by assigning them -∞ logit
        for bad_id in BAD_WORD_IDS:
            next_logits[bad_id] = float("-inf")
        # Batchify logits for processors (expects shape [batch_size, vocab_size])
        scores = next_logits.unsqueeze(0)              # shape (1, vocab_size)
        scores = processors(ids, scores)               # returns tensor (1, vocab_size)
        next_logits = scores[0]                        # back to shape (vocab_size,)

        probs = torch.softmax(next_logits, dim=-1)
        last_id = ids[0, -1].item()
        probs[last_id] = 0
        probs = probs / probs.sum()

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumulative <= top_p
            keep_mask[0] = True
            filtered_indices = sorted_indices[keep_mask]
            filtered_probs = sorted_probs[keep_mask] / sorted_probs[keep_mask].sum()
            next_id = filtered_indices[torch.multinomial(filtered_probs, 1)]
        else:
            vals, idxs = torch.topk(probs, k=top_k)
            vals = vals / vals.sum()
            next_id = idxs[torch.multinomial(vals, 1)]

        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
        token_str = tokenizer.decode([next_id.item()], clean_up_tokenization_spaces=False)
        print(token_str, end="", flush=True)
        if next_id.item() == tokenizer.eos_token_id:
            break


# --------------------------- CLI entry‑point ------------------------- #
def main():


    # Apply GPU memory fraction limit for CUDA if requested
    if DEVICE == "cuda" and gpu_percent is not None:
        fraction = gpu_percent / 100.0
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            print(f"Limiting CUDA memory usage to {gpu_percent}%")
        except Exception as e:
            print(f"Could not set CUDA memory fraction: {e}")
    # Note: PyTorch does not support GPU percentage throttling on MPS; this will only affect CUDA.

    # Apply MPS memory fraction limit when using Apple Silicon if requested
    if DEVICE == "mps" and gpu_percent is not None:
        fraction = gpu_percent / 100.0
        try:
            torch.mps.set_per_process_memory_fraction(fraction)
            print(f"Limiting MPS memory usage to {gpu_percent}%")
        except Exception as e:
            print(f"Could not set MPS memory fraction: {e}")

    dataset_choice = DATASET
    epochs_arg = str(EPOCHS)

    if dataset_choice == "tiny-shakespeare":
        tag = "tinyshakes"
    elif dataset_choice == "wikitext-103":
        tag = "wikitext103"
    elif dataset_choice == "bookcorpus":
        tag = "bookcorpus"
    else:
        tag = os.path.splitext(os.path.basename(dataset_choice))[0][:32]

    global MODEL_PATH
    MODEL_PATH = f"gpt_{tag}.pth"
    print(f"💾 Checkpoint path set to {MODEL_PATH}")
    # If using a directory dataset, store checkpoint inside that directory
    if os.path.isdir(dataset_choice):
        MODEL_PATH = os.path.join(dataset_choice, MODEL_PATH)
        print(f"🔄 Using checkpoint path: {MODEL_PATH}")

    # Load existing checkpoint if present
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        cfg = GPTConfig()
        model = GPT(cfg).to(DEVICE)  # ensure model exists before loading
        model.load_state_dict(state["state_dict"])
        print(f"🔄 Loaded checkpoint from {MODEL_PATH}")
    else:
        cfg = GPTConfig()
        model = GPT(cfg).to(DEVICE)

    if MODE == "train":

        def do_train():
            train_one_epoch(model, dataset_choice)

        if epochs_arg.lower() == "inf":
            epoch = 1
            while True:
                print(f"\n🔁 Epoch {epoch} (∞ mode)")
                do_train()
                epoch += 1
        else:
            num_epochs = int(epochs_arg)
            for epoch in range(1, num_epochs + 1):
                print(f"\n🔁 Epoch {epoch}/{num_epochs}")
                do_train()
    else:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("No checkpoint found; run train first.")
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state["state_dict"])
        print("🧠 Model loaded.")
        try:
            if PROMPT:
                generate_stream(model, PROMPT)
                print()
            else:
                while True:
                    prompt = input("\nYou> ")
                    generate_stream(model, prompt)
                    print()
        except KeyboardInterrupt:
            print("\n👋 Bye!")


if __name__ == "__main__":
    main()
