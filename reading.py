"""Reading: NikolAI reads children's stories and learns English from them by itself.

How: it reads a story word by word and always tries to guess the next word (a language model – the same principle
as in large language models, only very small: a GRU with about 3 million parameters). Nobody tells it what words
mean. It only learns which words appear in which contexts – "grab" and "take" show up in similar places, so they
end up close together in its head.

Then this reading brain becomes NikolAI's hearing and speaking. In its world it learns what the words refer to
(an apple, the forest). Hill et al. 2020 ("Human instruction-following with deep reinforcement learning via
transfer-learning from text") showed that an agent that read text first understands instructions with words its
teacher never used. That is exactly what Phase 6b tests ("Grab a stone!" instead of "Take a stone!").

Books (version 9): the BabyLM corpus (Warstadt et al. 2023) – about 100 million words, roughly what a child hears
by the age of twelve, all written or spoken by people: parents talking to small children (CHILDES), conversations
(British National Corpus, Switchboard), children's books (Project Gutenberg, Children's Book Test), film subtitles,
educational videos and Simple Wikipedia. Before, it read TinyStories (Eldan & Li 2023), which GPT wrote.

Pre-reading (once, before the world – takes hours):
    python -m neuro.reading --book ~/nikolai/books/babylm --reader ~/nikolai/neuro/reader_babylm.pt --hours 8
"""

import argparse
import collections
import os
import random
import re
import time

import torch

PAD, START, END, UNKNOWN = 0, 1, 2, 3
SPECIAL = ["<pad>", "<start>", "<end>", "<unk>"]
WORD_RE = re.compile(r"[a-z]+|[.!?]")
VOCAB = 8000              # the most frequent words in the books
RESERVE = 300             # room for new words it only hears in its world (e.g. "bast")
TEST_SHARE = 0.01         # the last 1 % of the book is never read – only for tests


def tokenize(text):
    """Text → words. Sentence ends become '.'; so do line breaks (BabyLM: one document or a few utterances per line)
    and the story separator of TinyStories."""
    text = text.replace("<|endoftext|>", " . ").replace("\n", " . ").lower()
    out = []
    for w in WORD_RE.findall(text):
        w = "." if w in "!?" else w
        if not (w == "." and out and out[-1] == "."):
            out.append(w)
    return out


class Book:
    """A (possibly huge) text file – or a folder of them (BabyLM: one file per source). Reads random pieces
    instead of loading everything. Bigger files are read more often, so every source keeps its share."""

    def __init__(self, file_path, rng=None):
        self.file_path = os.path.expanduser(file_path)
        if os.path.isdir(self.file_path):
            self.files = sorted(os.path.join(self.file_path, n) for n in os.listdir(self.file_path)
                                  if n.endswith((".txt", ".train")) and os.path.getsize(os.path.join(self.file_path, n)) > 10000)
        else:
            self.files = [self.file_path]
        if not self.files:
            raise FileNotFoundError(f"no .txt files in {self.file_path}")
        self.sizes = [os.path.getsize(d) for d in self.files]
        self.size = sum(self.sizes)
        self.z = rng or random.Random()
        self.f = [open(d, "rb") for d in self.files]

    def piece(self, words=80, test=False):
        """About `worte` consecutive words from a random place (in the test part – the last 1 % of every file –
        if test=True)."""
        while True:
            i = self.z.choices(range(len(self.files)), self.sizes)[0]
            size = self.sizes[i]
            cutoff = int(size * (1 - TEST_SHARE))
            a, b = (cutoff, size) if test else (0, cutoff)
            pos = self.z.randrange(a, max(a + 1, b - words * 12))
            self.f[i].seek(pos)
            raw = self.f[i].read(words * 12).decode("utf-8", "ignore")
            w = tokenize(raw)[1:]                 # the first word may be cut off
            if len(w) >= words:
                return w[:words]

    def word_counts(self, mb=40):
        """Word frequencies in about `mb` megabytes, spread over all files by size (for the vocabulary)."""
        c = collections.Counter()
        for d, g in zip(self.files, self.sizes):
            share = mb * 2 ** 20 * g / self.size
            with open(d, "rb") as f:
                words_read = 0
                while words_read < share:
                    raw = f.read(2 ** 20)
                    if not raw:
                        break
                    words_read += len(raw)
                    c.update(tokenize(raw.decode("utf-8", "ignore")))
        return c


def word_list(book, vocab=VOCAB):
    from . import teacher as L
    c = book.word_counts()
    words_list = SPECIAL + [w for w, _ in c.most_common(vocab)]
    for sentence in L.all_sentences():                 # the teacher's words must be there, even rare ones like "bast"
        for w in L.words(sentence):
            if w not in words_list:
                words_list.append(w)
    return words_list


def batch(book, ws, n=16, length=64, test=False):
    """n pieces of text as word numbers (B, laenge)."""
    return torch.tensor([[ws.index_of.get(w, UNKNOWN) for w in book.piece(length, test)] for _ in range(n)])


@torch.no_grad()
def perplexity(g, book, ws, pieces=40, seed=0):
    """Perplexity on text it never read: how many words is it on average 'unsure between'? Lower is better.
    Baseline: only knowing how frequent each word is (unigram)."""
    old = book.z
    book.z = random.Random(seed)
    x = batch(book, ws, pieces, 64, test=True)
    book.z = old
    error = float(g.reading_loss(x))
    counts = torch.bincount(x[:, 1:].reshape(-1), minlength=g.vocab).float() + 0.1
    p = counts / counts.sum()
    unigram = float(-(torch.log(p[x[:, 1:].reshape(-1)])).mean())
    return {"ppl": round(2.718281828 ** error, 1), "unigram": round(2.718281828 ** unigram, 1)}


def tell_story(g, ws, prompt="once upon a time there was a little", n=40, temperature=0.7, seed=None):
    z = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = [ws.index_of.get(w, UNKNOWN) for w in tokenize(prompt)]
    out = g.continue_text(ids, n, temperature, z)
    text = " ".join([prompt] + [ws.word(i) for i in out])
    return re.sub(r" \.", ".", text)


def pre_read(book_path, reader_path, hours=8.0, threads=6, length=64, n=32):
    """Pre-reading: only the reading part of the brain learns. Saves every 10 minutes; can be resumed."""
    from .brain import Brain, Vocabulary
    torch.set_num_threads(threads)
    book = Book(book_path)
    if os.path.exists(reader_path):
        from . import compat
        d = compat.migrate(torch.load(reader_path, weights_only=False))
        ws = Vocabulary(d["size"], d["vocab_list"])
        g = Brain(vocab=ws.size)
        g.load_state_dict(d["brain"], strict=False)
        words_read, history = d["words_read"], d["history"]
        reader_id = d.get("reader_id")
        print(f"continuing: {words_read:,} words read so far", flush=True)
    else:
        print("counting words …", flush=True)
        words_list = word_list(book)
        ws = Vocabulary(len(words_list) + RESERVE, words_list)
        g = Brain(vocab=ws.size)
        words_read, history = 0, []
        reader_id = f"{os.path.basename(book.file_path.rstrip('/'))}-{int(time.time())}"
        print(f"vocabulary: {len(words_list):,} words", flush=True)
    parts = g.reader_params()
    opt = torch.optim.Adam(parts, lr=2e-3)
    t0 = t_save = t_report = time.time()
    total_sum, k = 0.0, 0

    def save_brain():
        tmp = reader_path + ".tmp"
        torch.save({"brain": g.state_dict(), "vocab_list": ws.words_list, "size": ws.size, "words_read": words_read,
                    "history": history, "reader_id": reader_id}, tmp)
        os.replace(tmp, reader_path)

    try:
        while time.time() - t0 < hours * 3600:
            x = batch(book, ws, n, length)
            loss = g.reading_loss(x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parts, 1.0)
            opt.step()
            words_read += x.numel()
            total_sum, k = total_sum + float(loss.detach()), k + 1
            if time.time() - t_report >= 120:
                t_report = time.time()
                v = perplexity(g, book, ws)
                history.append({"words_read": words_read, "error": total_sum / k, **v})
                total_sum, k = 0.0, 0
                print(f"{(time.time() - t0) / 3600:5.2f} h | {words_read:>13,} words read | perplexity "
                      f"{v['ppl']} (word frequency only: {v['unigram']})\n        "
                      f"{tell_story(g, ws, seed=1)}", flush=True)
            if time.time() - t_save >= 600:
                t_save = time.time()
                save_brain()
    finally:
        save_brain()
        print("saved:", reader_path, flush=True)


def main():
    a = argparse.ArgumentParser(description="NikolAI reads: learns English from stories before entering its world")
    a.add_argument("--book", "--buch", dest="book", required=True, help="text file or folder of text files, e.g. BabyLM")
    a.add_argument("--reader", "--leser", dest="reader", default="~/nikolai/neuro/reader.pt", help="where the reading brain is saved")
    a.add_argument("--hours", "--stunden", dest="hours", type=float, default=8.0, help="how long to read")
    a.add_argument("--threads", type=int, default=6)
    args = a.parse_args()
    file_path = os.path.expanduser(args.reader)
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    import signal
    import sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    pre_read(args.book, file_path, args.hours, args.threads)


if __name__ == "__main__":
    main()
