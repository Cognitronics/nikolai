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
    python -m neuro.lesen --buch ~/nikolai/buecher/babylm --leser ~/nikolai/neuro/leser_babylm.pt --stunden 8
"""

import argparse
import collections
import os
import random
import re
import time

import torch
import torch.nn.functional as F

PAD, START, ENDE, UNBEKANNT = 0, 1, 2, 3
SONDER = ["<pad>", "<start>", "<ende>", "<unk>"]
WORT_RE = re.compile(r"[a-z]+|[.!?]")
VOKAB = 8000              # the most frequent words in the books
RESERVE = 300             # room for new words it only hears in its world (e.g. "bast")
TESTANTEIL = 0.01         # the last 1 % of the book is never read – only for tests


def zerlegen(text):
    """Text → words. Sentence ends become '.'; so do line breaks (BabyLM: one document or a few utterances per line)
    and the story separator of TinyStories."""
    text = text.replace("<|endoftext|>", " . ").replace("\n", " . ").lower()
    out = []
    for w in WORT_RE.findall(text):
        w = "." if w in "!?" else w
        if not (w == "." and out and out[-1] == "."):
            out.append(w)
    return out


class Buch:
    """A (possibly huge) text file – or a folder of them (BabyLM: one file per source). Reads random pieces
    instead of loading everything. Bigger files are read more often, so every source keeps its share."""

    def __init__(self, pfad, zufall=None):
        self.pfad = os.path.expanduser(pfad)
        if os.path.isdir(self.pfad):
            self.dateien = sorted(os.path.join(self.pfad, n) for n in os.listdir(self.pfad)
                                  if n.endswith((".txt", ".train")) and os.path.getsize(os.path.join(self.pfad, n)) > 10000)
        else:
            self.dateien = [self.pfad]
        if not self.dateien:
            raise FileNotFoundError(f"no .txt files in {self.pfad}")
        self.groessen = [os.path.getsize(d) for d in self.dateien]
        self.groesse = sum(self.groessen)
        self.z = zufall or random.Random()
        self.f = [open(d, "rb") for d in self.dateien]

    def stueck(self, worte=80, test=False):
        """About `worte` consecutive words from a random place (in the test part – the last 1 % of every file –
        if test=True)."""
        while True:
            i = self.z.choices(range(len(self.dateien)), self.groessen)[0]
            groesse = self.groessen[i]
            grenze = int(groesse * (1 - TESTANTEIL))
            a, b = (grenze, groesse) if test else (0, grenze)
            pos = self.z.randrange(a, max(a + 1, b - worte * 12))
            self.f[i].seek(pos)
            roh = self.f[i].read(worte * 12).decode("utf-8", "ignore")
            w = zerlegen(roh)[1:]                 # the first word may be cut off
            if len(w) >= worte:
                return w[:worte]

    def wortzaehlung(self, mb=40):
        """Word frequencies in about `mb` megabytes, spread over all files by size (for the vocabulary)."""
        c = collections.Counter()
        for d, g in zip(self.dateien, self.groessen):
            anteil = mb * 2 ** 20 * g / self.groesse
            with open(d, "rb") as f:
                gelesen = 0
                while gelesen < anteil:
                    roh = f.read(2 ** 20)
                    if not roh:
                        break
                    gelesen += len(roh)
                    c.update(zerlegen(roh.decode("utf-8", "ignore")))
        return c


def wortliste(buch, vokab=VOKAB):
    from . import lehrer as L
    c = buch.wortzaehlung()
    liste = SONDER + [w for w, _ in c.most_common(vokab)]
    for satz in L.alle_saetze():                 # the teacher's words must be there, even rare ones like "bast"
        for w in L.worte(satz):
            if w not in liste:
                liste.append(w)
    return liste


def stapel(buch, ws, n=16, laenge=64, test=False):
    """n pieces of text as word numbers (B, laenge)."""
    return torch.tensor([[ws.nr.get(w, UNBEKANNT) for w in buch.stueck(laenge, test)] for _ in range(n)])


@torch.no_grad()
def verwirrung(g, buch, ws, stuecke=40, seed=0):
    """Perplexity on text it never read: how many words is it on average 'unsure between'? Lower is better.
    Baseline: only knowing how frequent each word is (unigram)."""
    alt = buch.z
    buch.z = random.Random(seed)
    x = stapel(buch, ws, stuecke, 64, test=True)
    buch.z = alt
    fehler = float(g.lesen_verlust(x))
    zaehl = torch.bincount(x[:, 1:].reshape(-1), minlength=g.vokab).float() + 0.1
    p = zaehl / zaehl.sum()
    unigramm = float(-(torch.log(p[x[:, 1:].reshape(-1)])).mean())
    return {"perplexitaet": round(2.718281828 ** fehler, 1), "unigramm": round(2.718281828 ** unigramm, 1)}


def erzaehlen(g, ws, anfang="once upon a time there was a little", n=40, temperatur=0.7, seed=None):
    z = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = [ws.nr.get(w, UNBEKANNT) for w in zerlegen(anfang)]
    out = g.weiterlesen(ids, n, temperatur, z)
    text = " ".join([anfang] + [ws.wort(i) for i in out])
    return re.sub(r" \.", ".", text)


def vorlesen(buch_pfad, leser_pfad, stunden=8.0, threads=6, laenge=64, n=32):
    """Pre-reading: only the reading part of the brain learns. Saves every 10 minutes; can be resumed."""
    from .gehirn import Gehirn, Wortschatz
    torch.set_num_threads(threads)
    buch = Buch(buch_pfad)
    if os.path.exists(leser_pfad):
        d = torch.load(leser_pfad, weights_only=False)
        ws = Wortschatz(d["groesse"], d["woerter"])
        g = Gehirn(vokab=ws.groesse)
        g.load_state_dict(d["gehirn"], strict=False)
        gelesen, verlauf = d["gelesen"], d["verlauf"]
        leser_id = d.get("leser_id")
        print(f"continuing: {gelesen:,} words read so far", flush=True)
    else:
        print("counting words …", flush=True)
        liste = wortliste(buch)
        ws = Wortschatz(len(liste) + RESERVE, liste)
        g = Gehirn(vokab=ws.groesse)
        gelesen, verlauf = 0, []
        leser_id = f"{os.path.basename(buch.pfad.rstrip('/'))}-{int(time.time())}"
        print(f"vocabulary: {len(liste):,} words", flush=True)
    teile = g.leser_teile()
    opt = torch.optim.Adam(teile, lr=2e-3)
    t0 = t_speichern = t_bericht = time.time()
    summe, k = 0.0, 0

    def speichern():
        tmp = leser_pfad + ".tmp"
        torch.save({"gehirn": g.state_dict(), "woerter": ws.liste, "groesse": ws.groesse, "gelesen": gelesen,
                    "verlauf": verlauf, "leser_id": leser_id}, tmp)
        os.replace(tmp, leser_pfad)

    try:
        while time.time() - t0 < stunden * 3600:
            x = stapel(buch, ws, n, laenge)
            verlust = g.lesen_verlust(x)
            opt.zero_grad()
            verlust.backward()
            torch.nn.utils.clip_grad_norm_(teile, 1.0)
            opt.step()
            gelesen += x.numel()
            summe, k = summe + float(verlust.detach()), k + 1
            if time.time() - t_bericht >= 120:
                t_bericht = time.time()
                v = verwirrung(g, buch, ws)
                verlauf.append({"gelesen": gelesen, "fehler": summe / k, **v})
                summe, k = 0.0, 0
                print(f"{(time.time() - t0) / 3600:5.2f} h | {gelesen:>13,} words read | perplexity "
                      f"{v['perplexitaet']} (word frequency only: {v['unigramm']})\n        "
                      f"{erzaehlen(g, ws, seed=1)}", flush=True)
            if time.time() - t_speichern >= 600:
                t_speichern = time.time()
                speichern()
    finally:
        speichern()
        print("saved:", leser_pfad, flush=True)


def main():
    a = argparse.ArgumentParser(description="NikolAI reads: learns English from stories before entering its world")
    a.add_argument("--buch", required=True, help="text file or folder of text files, e.g. BabyLM")
    a.add_argument("--leser", default="~/nikolai/neuro/leser.pt", help="where the reading brain is saved")
    a.add_argument("--stunden", type=float, default=8.0, help="how long to read")
    a.add_argument("--threads", type=int, default=6)
    args = a.parse_args()
    pfad = os.path.expanduser(args.leser)
    os.makedirs(os.path.dirname(os.path.abspath(pfad)), exist_ok=True)
    import signal
    import sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    vorlesen(args.buch, pfad, args.stunden, args.threads)


if __name__ == "__main__":
    main()
