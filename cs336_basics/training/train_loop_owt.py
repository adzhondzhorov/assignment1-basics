import datetime
from pathlib import Path
import time

from cs336_basics.layers.softmax import softmax
from cs336_basics.layers.transformer_lm import TransformerLM
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.training.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.training.data_loader import load_batches
from cs336_basics.training.loss import cross_entropy
from cs336_basics.training.adamw import AdamW


import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# import wandb


seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DATASET = "owt"

RAW_DATA_MAP = {
    "tiny_stories": {
        "train": "data/TinyStoriesV2-GPT4-train.txt",
        "val": "data/TinyStoriesV2-GPT4-valid.txt",
    },
    "owt": {
        "train": "data/owt_train.txt",
        "val": "data/owt_valid.txt",
    },
}

SPECIAL_TOKENS = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(
    f"storage/tokenizers/{DATASET}/vocab.pkl",
    f"storage/tokenizers/{DATASET}/merges.pkl",
    SPECIAL_TOKENS,
)

VOCAB_SIZE = len(tokenizer.vocab)
CONTEXT_LEN = 512
# DEVICE = "cuda:0"
DEVICE = "mps"


log_file = open(f"storage/logs/{DATASET}_output.log", "a")


def print_log(*values, sep=" ", end="\n"):
    print(*values, sep=sep, end=end, flush=True)
    print(*values, sep=sep, end=end, file=log_file, flush=True)


def build_loss_curve(losses):
    base_path = "storage/loss_curves/"
    np.save(base_path + f"{DATASET}_losses.np", losses)
    train_losses, val_losses = zip(*losses)
    x = range(1, len(losses) + 1)

    plt.plot(x, train_losses, label="train loss")
    plt.plot(x, val_losses, "o", label="val loss")

    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.grid()
    plt.legend()

    plt.savefig(base_path + f"{DATASET}.png", dpi=300, bbox_inches="tight")
    plt.show()


def load_tokens(set_name):
    tokens_path = f"storage/tokenized_sets/{DATASET}_{set_name}.npy"
    path = Path(tokens_path)

    if not path.exists():
        with open(RAW_DATA_MAP[DATASET][set_name], "rb") as f:
            text = f.read()
            tokens = tokenizer.encode(text)
            del text
            np.save(tokens_path, tokens)
    return np.load(tokens_path)


def load_val_set():
    val_tokens = load_tokens("val")
    val_set = [], []
    n_windows = len(val_tokens) // CONTEXT_LEN
    for s in range(n_windows):
        for i in range(2):
            val_set[i].append(
                val_tokens[s * CONTEXT_LEN + i : (s + 1) * CONTEXT_LEN + i]
            )
    val_set = torch.tensor(val_set).to(DEVICE)
    return val_set


def eval_val_set(llm, val_set, batch_size):
    llm.eval()
    with torch.no_grad():
        loss = 0
        batches_len = len(val_set[0]) // batch_size
        for b in range(batches_len):
            batch = val_set[:, b * batch_size : (b + 1) * batch_size]
            logits = llm.forward(batch[0])
            loss += cross_entropy(logits, batch[1])
        loss = loss / batches_len
    llm.train()
    val_loss = loss.cpu().item()
    return val_loss


def build_transformer_lm():
    llm = TransformerLM(
        VOCAB_SIZE,
        context_len=CONTEXT_LEN,
        num_layers=4,
        d_model=512,
        d_ffn=1344,
        num_heads=16,
        rope_theta=10000.0,
        device=DEVICE,
    )
    llm = torch.compile(llm)
    return llm


def train_loop(tokens, llm, optimizer, batch_size):
    train_batches, test_batches = load_batches(tokens, batch_size, CONTEXT_LEN, DEVICE)
    optimizer.zero_grad()
    logits = llm.forward(train_batches)
    loss = cross_entropy(logits, test_batches)
    loss.backward()
    optimizer.step()
    train_loss = loss.cpu().item()
    return train_loss


def train(optimizer_params, iterations, batch_size):
    train_tokens = load_tokens("train")
    val_set = load_val_set()

    llm = build_transformer_lm()

    optimizer = AdamW(llm.parameters(), **optimizer_params)

    losses = []

    # tracker = wandb.init(project="stanford-cs336", name="assignment1-basics")

    start_time = time.perf_counter()

    for i in range(1, iterations + 1):
        train_loss = train_loop(train_tokens, llm, optimizer, batch_size)

        if i % 10 == 0:
            elapsed_time = time.perf_counter() - start_time
            print_log(
                "train loss {}: {:.4f} {:.4f}s".format(i, train_loss, elapsed_time)
            )
            # tracker.log({"train_loss": float(train_loss),"elapsed_seconds": time.perf_counter() - start_time }, step=i)

        val_loss = None
        if i % 13000 == 0:
            val_loss = eval_val_set(llm, val_set, batch_size)
            print_log("val loss {}: {:.4f}".format(i, val_loss))
            # tracker.log({"val_loss": float(val_loss), "elapsed_seconds": time.perf_counter() - start_time }, step=i)

        losses.append((train_loss, val_loss))
    # tracker.finish()
    build_loss_curve(losses)
    return llm


def load(iteration):
    llm = build_transformer_lm()
    load_checkpoint(f"storage/checkpoints/{DATASET}_{iteration}.chp", llm, None)
    return llm


def generate(llm, prompt, max_len=50):
    tokens = tokenizer.encode(prompt)
    for i in range(max_len):
        logits = llm(torch.tensor([tokens], dtype=torch.int64))
        all_probs = softmax(logits, -1)
        seq_probs = all_probs[0]  # 1st and only batch
        probs = seq_probs[-1]  # last sequence
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = tokens + [next_token.item()]
        if next_token == 0:
            break
    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    ITERS = 13000

    print_log("======================")
    print_log(str(datetime.datetime.now()))
    for optimizer_params in [
        dict(
            alpha=None,
            lr_schedule=dict(
                alpha_max=3e-3, alpha_min=5e-6, t_warmup=100, t_final=ITERS
            ),
            lambda_=0.1,
            beta1=0.9,
            beta2=0.975,
            epsilon=1e-8,
        ),
    ]:
        print_log(optimizer_params)
        llm = train(optimizer_params, iterations=ITERS, batch_size=32)
