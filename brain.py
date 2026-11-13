"""The neural brain. At first all connections are random – it knows nothing, not a single word.

Structure (everything learns together, nothing is pre-programmed except the senses):
  senses ──► perception ─┐
  heard  ──► hearing ────┼─► memory (GRU) ─► acting (verb, thing) and value (how good is the situation?)
  last action ───────────┘          │
                                    ├─► world model: "What will I sense next?" – its error is the curiosity
                                    ├─► speaking: says in words what just happened
                                    └─► talking (version 3): a switch decides each step whether it also says
                                        something (up to five words) – talking and acting at the same time, as
                                        children do. It is learned in two ways: repeating what the teacher models
                                        ("Say: I want an apple"), and reward when what it says achieves something.

Version 7 – reading: hearing and speaking get a language memory from books (reading.py). Hearing is a language
model (GRU) that learned to predict the next word while reading. Speaking and talking start with the same
weights – so NikolAI can already "tell stories" before it knows what the words refer to.

To the brain, words are just numbers at first. What "apple" means, it has to find out by itself –
from how what it hears, sees, does and is rewarded for fits together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import OBJECTS, SENSES, VERBS

PAD, START, END = 0, 1, 2
TALK_MAX = 8            # up to 7 words + end ("there is an apple in the forest")
WORD_DIM = 128            # how many numbers a word has in its head
READ_H = 256              # memory while reading (same size as the main memory, so speaking can take it over)


class Vocabulary:
    """Gives heard words numbers – in the order in which they are first heard."""

    def __init__(self, size=400, words_list=None):
        self.size = size
        self.words_list = list(words_list or ["<pad>", "<start>", "<end>"])
        self.index_of = {w: i for i, w in enumerate(self.words_list)}

    def encode_words(self, words, maxlen=16):
        out = []
        for w in words[:maxlen]:
            if w not in self.index_of and len(self.words_list) < self.size:
                self.index_of[w] = len(self.words_list)
                self.words_list.append(w)
            if w in self.index_of:
                out.append(self.index_of[w])
        return out

    def word(self, i):
        return self.words_list[i] if i < len(self.words_list) else "?"


class Brain(nn.Module):
    def __init__(self, vocab=400, h=256):
        super().__init__()
        self.h = h
        nv, no = len(VERBS), len(OBJECTS)
        self.nv, self.no = nv, no
        self.perception = nn.Sequential(nn.Linear(SENSES, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
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
        self.world_model = nn.Sequential(nn.Linear(h + 32, 256), nn.ReLU(), nn.Linear(256, SENSES))
        self.speak_start = nn.Linear(h + 128 + 32, h)
        self.speaking = nn.GRU(WORD_DIM, h, batch_first=True)
        self.speak_out = nn.Linear(h, vocab)
        self.vocab = vocab
        self.talk_cell = nn.GRUCell(WORD_DIM, h)
        self.talk_out = nn.Linear(h, vocab)
        self.speak_switch = nn.Linear(h, 1)
        nn.init.constant_(self.speak_switch.bias, -2.0)       # rather quiet at first

    def no_action(self, b):
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

    # ------------------------------------------------------------ reading
    def reader_params(self):
        return list(self.word.parameters()) + list(self.hearing.parameters()) + list(self.read_out.parameters())

    def reading_loss(self, x):
        """x: (B, L) word numbers from a book. Error when guessing each next word."""
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
            logits[:, :4] = -1e9                                  # no special tokens
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            out.append(int(nxt))
            outp, hz = self.hearing(self.word(nxt), hz)
            logits = self.read_out(outp[:, -1])
        return out

    def from_reader(self):
        """Speaking and talking start with what reading has learned."""
        with torch.no_grad():
            for goal in (self.speaking,):
                for k, v in self.hearing.state_dict().items():
                    goal.state_dict()[k].copy_(v)
            for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(self.talk_cell, k).copy_(getattr(self.hearing, k + "_l0"))
            for goal in (self.speak_out, self.talk_out):
                goal.weight.copy_(self.read_out.weight)
                goal.bias.copy_(self.read_out.bias)

    def tick(self, senses, sentences, last_action, h):
        x = torch.cat([self.perception(senses), self.hear(sentences), self.act(last_action)], 1)
        return self.memory(x, h)

    def decide(self, h, verb=None):
        """Returns (verb, object distribution, verb distribution). With verb=None the verb is sampled."""
        vl = self.verb_head(h)
        if self.nv > 6:
            vl = vl.clone()
            vl[:, 6] = -1e9                     # "say" is no longer an action of its own: talking happens alongside
        vd = torch.distributions.Categorical(logits=vl)
        if verb is None:
            verb = vd.sample()
        ol = self.obj_head(torch.cat([h, self.verb_emb(verb)], 1))
        od = torch.distributions.Categorical(logits=ol)
        return verb, od, vd

    def talk(self, h, known, words=None):
        """Talking as an action: samples up to TALK_MAX words (only heard ones). Returns (words, logp, entropy).
        With words=(B, TALK_MAX) only the probability of these words is computed (for learning)."""
        b = h.shape[0]
        mask = torch.full((self.vocab,), -1e9)
        mask[4:max(5, known)] = 0                     # 3 = <unk>: never say it
        mask[END] = 0
        z = h
        inp_word = torch.full((b,), START, dtype=torch.long)
        active = torch.ones(b, dtype=torch.bool)
        logp, ent = torch.zeros(b), torch.zeros(b)
        out = torch.full((b, TALK_MAX), PAD, dtype=torch.long)
        for k in range(TALK_MAX):
            z = self.talk_cell(self.word(inp_word), z)
            logits = self.talk_out(z) + mask
            if k == 0:
                logits = logits.clone()
                logits[:, END] = -1e9                     # at least one word
            d = torch.distributions.Categorical(logits=logits)
            w = d.sample() if words is None else torch.where(active, words[:, k], torch.full_like(words[:, k], END))
            logp = logp + torch.where(active, d.log_prob(w), torch.zeros(b))
            ent = ent + torch.where(active, d.entropy(), torch.zeros(b))
            out[:, k] = torch.where(active, w, torch.full_like(w, PAD))
            active = active & (w != END)
            inp_word = torch.where(active, w, torch.full_like(w, END))
        return out, logp, ent

    def talk_switch(self, h):
        return torch.distributions.Bernoulli(logits=self.speak_switch(h).squeeze(1))

    def act_id(self, verb, obj):
        return verb * self.no + obj

    def predict(self, h, act):
        return self.world_model(torch.cat([h, self.act(act)], 1))

    def speak_state(self, h, senses_after, act):
        return torch.tanh(self.speak_start(torch.cat([h, self.perception(senses_after), self.act(act)], 1)))

    def speak_loss(self, z, goal):
        """z: (B, h) start state; goal: (B, L) word numbers without START, with END. Mean error per word."""
        inp = torch.cat([torch.full((goal.shape[0], 1), START, dtype=torch.long), goal[:, :-1]], 1)
        outp, _ = self.speaking(self.word(inp), z.unsqueeze(0))
        logits = self.speak_out(outp)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), goal.reshape(-1), ignore_index=PAD)

    def adopt(self, old):
        """Take over an older brain: everything learned stays, new parts start random."""
        new_sd = self.state_dict()
        nv_old = old["verb_head.weight"].shape[0]
        no = self.no
        for k, v in old.items():
            if k not in new_sd:
                continue
            if new_sd[k].shape == v.shape:
                new_sd[k] = v
            elif k in ("verb_head.weight", "verb_head.bias", "verb_emb.weight"):
                new_sd[k][:v.shape[0]] = v
            elif k == "act.weight":
                new_sd[k][:nv_old * no] = v[:nv_old * no]
                new_sd[k][self.nv * no] = v[nv_old * no]          # "no action yet" now sits further back
        missing = [k for k in new_sd if k not in old]
        self.load_state_dict(new_sd)
        return nv_old != self.nv or bool(missing)

    @torch.no_grad()
    def speak(self, z, maxlen=12, temperature=0.0):
        """Speaks from state z (1, h) – word by word, until the end token."""
        w = torch.tensor([[START]])
        hz = z.unsqueeze(0)
        out = []
        for _ in range(maxlen):
            outp, hz = self.speaking(self.word(w), hz)
            logits = self.speak_out(outp[:, -1])
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
