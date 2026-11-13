"""The neural brain. At first all connections are random – it knows nothing, not a single word.

Structure (everything learns together, nothing is pre-programmed except the senses):
  senses ──► perception ─┐
  heard ──► hearing ─────┼─► memory (GRU) ─► acting (verb, thing) and value (how good is the situation?)
  last action ───────────┘          │
                                    ├─► world model: "what will I sense next?" – its error is the curiosity
                                    ├─► speech: says in words what just happened (and answers questions)
                                    ├─► talking (version 3): a switch decides each step whether it also says
                                    │   something (up to seven words) – talking and acting at once, as children do.
                                    │   Learned in two ways: repeating what the teacher prompts ("Say: I want an
                                    │   apple") and reward when what it says achieves something.
                                    ├─► imagination (version 15): from heard words to expected properties
                                    └─► which one (version 17): any, a big one or a small one – the hands choose

Version 7 – reading: hearing and speaking get a language memory from books (reading.py). Hearing is a language model
(GRU) that learned to predict the next word while reading. Speech and talking start with the same weights – so
NikolAI can already "tell stories" before it knows what the words refer to.

At first, words are just numbers to the brain. What "apple" means it has to find out by itself – from what it
hears, sees, does, and the rewards that follow.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import MODS, OBJECTS, SENSES, VARIANTS, VERBS

PAD, START, END = 0, 1, 2
SPEECH_MAX = 8            # up to 7 words + end ("there is an apple in the forest")
WORD_DIM = 128            # how many numbers a word has in its head
READ_H = 256              # memory while reading (same size as the main memory, so speech can take it over)


class Vocabulary:
    """Gives heard words numbers – in the order in which they are first heard."""

    def __init__(self, capacity=400, vocab_list=None):
        self.capacity = capacity
        self.vocab_list = list(vocab_list or ["<pad>", "<start>", "<ende>"])
        self.index = {w: i for i, w in enumerate(self.vocab_list)}

    def encode_ids(self, words, maxlen=16):
        out = []
        for w in words[:maxlen]:
            if w not in self.index and len(self.vocab_list) < self.capacity:
                self.index[w] = len(self.vocab_list)
                self.vocab_list.append(w)
            if w in self.index:
                out.append(self.index[w])
        return out

    def word(self, i):
        return self.vocab_list[i] if i < len(self.vocab_list) else "?"


class Brain(nn.Module):
    def __init__(self, vocab=400, h=256):
        super().__init__()
        self.h = h
        nv, no = len(VERBS), len(OBJECTS)
        self.nv, self.no = nv, no
        self.perception = nn.Sequential(nn.Linear(SENSES, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        # imagination (version 15): from a heard word to expected properties – mean and uncertainty
        from .imagination import P as _P
        self.imagination = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2 * _P))
        with torch.no_grad():
            self.imagination[2].bias[_P:] = math.log(0.25)        # am Anfang: weite Unsicherheit = Nebel
        self.word = nn.Embedding(vocab, WORD_DIM, padding_idx=PAD)
        self.hearing = nn.GRU(WORD_DIM, READ_H, batch_first=True)      # = the reader
        self.read_out = nn.Linear(READ_H, vocab)                    # reading: which word comes next?
        self.hear_proj = nn.Linear(READ_H, 128)
        self.act = nn.Embedding(nv * no + 1, 32)                 # last action (+1: none yet)
        self.memory = nn.GRUCell(128 + 128 + 32, h)
        self.verb_head = nn.Linear(h, nv)
        self.verb_emb = nn.Embedding(nv, 32)
        self.obj_head = nn.Sequential(nn.Linear(h + 32, 128), nn.ReLU(), nn.Linear(128, no))
        self.value = nn.Linear(h, 1)
        # version 17: which one – any, a big one, a small one (only matters when taking, putting down or eating)
        self.mod_head = nn.Sequential(nn.Linear(h + 32, 64), nn.ReLU(), nn.Linear(64, len(MODS)))
        with torch.no_grad():
            self.mod_head[2].bias.copy_(torch.tensor([2.0, 0.0, 0.0]))     # at first: just any
        # version 18: how much the hands trust imagination ("which one fits what I expect?")
        self.trust = nn.Parameter(torch.tensor(1.0))
        self.world_model = nn.Sequential(nn.Linear(h + 32, 256), nn.ReLU(), nn.Linear(256, SENSES))
        self.speech_start = nn.Linear(h + 128 + 32, h)
        self.speech = nn.GRU(WORD_DIM, h, batch_first=True)
        self.speech_out = nn.Linear(h, vocab)
        self.vocab = vocab
        self.speak_cell = nn.GRUCell(WORD_DIM, h)
        self.speak_out = nn.Linear(h, vocab)
        self.talk_switch = nn.Linear(h, 1)
        nn.init.constant_(self.talk_switch.bias, -2.0)       # anfangs eher still

    def no_act(self, b):
        return torch.full((b,), self.nv * self.no, dtype=torch.long)

    def hear(self, sentences):
        """sentences: (B, L) word numbers, 0 = nothing. Returns (B, 128); silence = zeros."""
        if sentences.shape[1] == 0:
            return torch.zeros(sentences.shape[0], 128)
        lengths = (sentences != PAD).sum(1)
        outp, _ = self.hearing(self.word(sentences))
        idx = (lengths - 1).clamp(min=0)
        last_out = torch.tanh(self.hear_proj(outp[torch.arange(sentences.shape[0]), idx]))
        return last_out * (lengths > 0).float().unsqueeze(1)

    # ------------------------------------------------------------ Vorstellung
    def imagine(self, sentences):
        """sentences: (B, L) – a heard word like "apple". Returns (mean, log variance) of the properties."""
        outp = self.imagination(self.hear(sentences))
        p = outp.shape[1] // 2
        return torch.sigmoid(outp[:, :p]), outp[:, p:].clamp(-7, 1)

    # ------------------------------------------------------------ Lesen
    def reader_params(self):
        return list(self.word.parameters()) + list(self.hearing.parameters()) + list(self.read_out.parameters())

    def reading_loss(self, x):
        """x: (B, L) word numbers from a book. Error at guessing each next word."""
        outp, _ = self.hearing(self.word(x[:, :-1]))
        logits = self.read_out(outp)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=PAD)

    @torch.no_grad()
    def continue_text(self, ids, n=40, temperature=0.7, generator=None):
        """Continues a story (with the reader only)."""
        outp, hz = self.hearing(self.word(torch.tensor([ids])))
        logits = self.read_out(outp[:, -1])
        out = []
        for _ in range(n):
            logits[:, :4] = -1e9                                  # keine Sonderzeichen
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            out.append(int(nxt))
            outp, hz = self.hearing(self.word(nxt), hz)
            logits = self.read_out(outp[:, -1])
        return out

    def from_reader(self):
        """Speech and talking start with what reading has learned."""
        with torch.no_grad():
            for target in (self.speech,):
                for k, v in self.hearing.state_dict().items():
                    target.state_dict()[k].copy_(v)
            for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(self.speak_cell, k).copy_(getattr(self.hearing, k + "_l0"))
            for target in (self.speech_out, self.speak_out):
                target.weight.copy_(self.read_out.weight)
                target.bias.copy_(self.read_out.bias)

    def think(self, senses, sentences, last_act, h):
        x = torch.cat([self.perception(senses), self.hear(sentences), self.act(last_act)], 1)
        return self.memory(x, h)

    def decide(self, h, verb=None):
        """Returns (verb, obj, logp, entropy). With verb=None it samples."""
        vl = self.verb_head(h)
        if self.nv > 6:
            vl = vl.clone()
            vl[:, 6] = -1e9                     # "say" as an action of its own is gone: talking happens alongside
        vd = torch.distributions.Categorical(logits=vl)
        if verb is None:
            verb = vd.sample()
        ol = self.obj_head(torch.cat([h, self.verb_emb(verb)], 1))
        od = torch.distributions.Categorical(logits=ol)
        return verb, od, vd

    def choose(self, h, verb, match=None):
        """Which one the hands choose (any / big / small). match (version 18): how much better the big one fits what
        it expects than the small one – pushes big or small, as much as it trusts its imagination."""
        logits = self.mod_head(torch.cat([h, self.verb_emb(verb)], 1))
        if match is not None:
            logits = logits + self.trust * match.unsqueeze(1) * torch.tensor([0.0, 0.5, -0.5])
        return torch.distributions.Categorical(logits=logits)

    def speak(self, h, known, words=None):
        """Talking as an action: samples up to SPEECH_MAX words (only heard ones). Returns (words, logp, entropy).
        With words=(B, SPEECH_MAX) it only computes the probability of these words (for learning)."""
        b = h.shape[0]
        mask = torch.full((self.vocab,), -1e9)
        mask[4:max(5, known)] = 0                     # 3 = <unk>: nie sagen
        mask[END] = 0
        z = h
        inp = torch.full((b,), START, dtype=torch.long)
        active = torch.ones(b, dtype=torch.bool)
        logp, ent = torch.zeros(b), torch.zeros(b)
        out = torch.full((b, SPEECH_MAX), PAD, dtype=torch.long)
        for k in range(SPEECH_MAX):
            z = self.speak_cell(self.word(inp), z)
            logits = self.speak_out(z) + mask
            if k == 0:
                logits = logits.clone()
                logits[:, END] = -1e9                     # at least one word
            d = torch.distributions.Categorical(logits=logits)
            w = d.sample() if words is None else torch.where(active, words[:, k], torch.full_like(words[:, k], END))
            logp = logp + torch.where(active, d.log_prob(w), torch.zeros(b))
            ent = ent + torch.where(active, d.entropy(), torch.zeros(b))
            out[:, k] = torch.where(active, w, torch.full_like(w, PAD))
            active = active & (w != END)
            inp = torch.where(active, w, torch.full_like(w, END))
        return out, logp, ent

    def switch(self, h):
        return torch.distributions.Bernoulli(logits=self.talk_switch(h).squeeze(1))

    def act_index(self, verb, obj):
        return verb * self.no + obj

    def expectation(self, h, act):
        return self.world_model(torch.cat([h, self.act(act)], 1))

    def speech_state(self, h, senses_after, act):
        return torch.tanh(self.speech_start(torch.cat([h, self.perception(senses_after), self.act(act)], 1)))

    def speech_loss(self, z, target):
        """z: (B, h) start state; target: (B, L) word numbers without START, with END. Mean error per word."""
        inp_ = torch.cat([torch.full((target.shape[0], 1), START, dtype=torch.long), target[:, :-1]], 1)
        outp, _ = self.speech(self.word(inp_), z.unsqueeze(0))
        logits = self.speech_out(outp)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD)

    def adopt(self, old):
        """Take over an older brain: everything learned stays, new parts start random."""
        fresh = self.state_dict()
        nv_old = old["verb_head.weight"].shape[0]
        no = self.no
        for k, v in old.items():
            if k not in fresh:
                continue
            if fresh[k].shape == v.shape:
                fresh[k] = v
            elif k in ("verb_head.weight", "verb_head.bias", "verb_emb.weight"):
                fresh[k][:v.shape[0]] = v
            elif k == "act.weight":
                fresh[k][:nv_old * no] = v[:nv_old * no]
                fresh[k][self.nv * no] = v[nv_old * no]          # "no action yet" now sits further back
            else:
                fresh[k] = _overlap(fresh[k], v)
        missing = [k for k in fresh if k not in old]
        self.load_state_dict(fresh)
        return nv_old != self.nv or bool(missing)

    def load_grown(self, old):
        """Load a brain from an older version whose senses were fewer (version 17 added sizes): what it knows stays,
        new senses start with no influence, new parts start fresh. Returns the names of new parts."""
        fresh = self.state_dict()
        for k, v in old.items():
            if k in fresh:
                fresh[k] = v if fresh[k].shape == v.shape else _overlap(fresh[k], v)
        self.load_state_dict(fresh)
        return sorted({k.split(".")[0] for k in fresh if k not in old})

    @torch.no_grad()
    def utter(self, z, maxlen=12, temperature=0.0):
        """Speaks from state z (1, h) – word by word, up to the end sign."""
        w = torch.tensor([[START]])
        hz = z.unsqueeze(0)
        out = []
        for _ in range(maxlen):
            outp, hz = self.speech(self.word(w), hz)
            logits = self.speech_out(outp[:, -1])
            logits[:, [PAD, START, 3]] = -1e9                    # 3 = <unk> (with reading) – never said
            if temperature > 0:
                nxt = torch.distributions.Categorical(logits=logits / temperature).sample()
            else:
                nxt = logits.argmax(-1)
            n = int(nxt)
            if n == END:
                break
            out.append(n)
            w = nxt.view(1, 1)
        return out


def relevant(verb, obj):
    """Does "which one" matter for this action? (taking, putting down or eating an apple, a branch or a stone)"""
    vs = torch.tensor([VERBS.index(x) for x in ("take", "put", "eat")])
    os_ = torch.tensor([OBJECTS.index(x) for x in VARIANTS])
    return (verb.unsqueeze(1) == vs).any(1) & (obj.unsqueeze(1) == os_).any(1)


def _overlap(new, old):
    """Copy an older, smaller weight into the new one where they overlap. New inputs get zero weight (they change
    nothing at first), new outputs keep their fresh start."""
    out = new.clone()
    if new.dim() == 2 and new.shape[0] == old.shape[0] and new.shape[1] > old.shape[1]:
        out[:, old.shape[1]:] = 0.0                       # new senses: no influence yet
    sl = tuple(slice(0, min(a, b)) for a, b in zip(new.shape, old.shape))
    out[sl] = old[sl]
    return out
