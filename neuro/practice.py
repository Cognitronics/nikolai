"""Practice: the brain lives in many copies of the world at once (like many hours of practice at a time) and learns.

Learning signals:
  * Body: hunger hurts, getting full feels good.
  * Curiosity: what the world model did not predict is exciting (reward = surprise, Pathak et al. 2017). The better
    the world model gets, the more boring the familiar becomes – the creature looks for new things.
  * Teacher: praise when an instruction is fulfilled.
  * Speech: the brain tries to predict what the teacher is about to say about its action.
  * Hindsight (version 6): after every successful action, practice asks: which instruction would this have
    fulfilled? ("You took a stone" → "Take a stone!"). Then it replays the last steps as if it had heard that
    sentence at the start, and learns: whoever hears this sentence does this. So every action becomes a lesson
    (Andrychowicz et al. 2017, "Hindsight Experience Replay"; Ghosh et al. 2019, GCSL; Chan et al. 2019).
  * Answers (version 8): when the teacher asks "Where are you?", speech learns the true answer in the next step
    ("i am in the forest") – from what it just heard and how it is doing.
  * Friends (version 10): the worlds are shared in pairs – NikolAI and Kogli (one brain, two bodies, like twins).
    Each hears what the other says and feels it when the other gets full (a shared reward, as with cooperating
    agents in research, e.g. Foerster et al. 2016, Lowe et al. 2017).
  * Reading (version 7): alongside, it keeps reading its book (guessing the next word), so hearing doesn't forget
    what it knows from books while it learns what the words refer to in its world.
  * Imagination (version 15): what it perceives about things (seeing, taking, carrying, eating) trains its
    expectations of them – see imagination.py.
Method: PPO (Schulman et al. 2017) with GAE; world model, speech and the rest are trained at the same time.
"""

import math
import random
import time

import torch

from . import legacy
import torch.nn.functional as F

from . import teacher as L
from . import reading
from . import imagination as V
from .brain import END, PAD, SPEECH_MAX, Brain, Vocabulary, relevant
from .world import VERBS, World

SAY = VERBS.index("say")


def words_from(ws, row):
    return [ws.word(int(i)) for i in row if int(i) not in (PAD, END)]

GAMMA, LAMBDA = 0.97, 0.95
CURIOSITY = 0.05
EPOCHS, CLIP = 3, 0.2
HINDSIGHT_MAX = 5               # the sentence is put in up to this many steps before the action, "after the fact"
HINDSIGHT_WEIGHT = 0.3
READING_WEIGHT = 0.5
PERCEPTIONS_MAX = 256         # new ones per practice section
IMAGINATION_BUFFER = 4000       # this many experiences with things are remembered
IMAGINATION_STEPS = 8        # learning steps of the imagination per section
TEAM = 0.5                      # how much it shares in it when the friend gets full
LANGUAGE_PARTS = ("word.", "hearing.", "read_out.", "hear_proj.", "speech.", "speech_out.", "speech_start.",
                  "speak_cell.", "speak_out.")
SIZE_WEIGHT = 0.5               # version 17: lessons for the hands (which one)
SPEECH_COST = 0.04              # talking costs a little effort – otherwise it chatters nonstop


def sentence_tensor(lists, maxlen=24):
    n = max([len(x) for x in lists] + [0])
    t = torch.zeros(len(lists), min(n, maxlen), dtype=torch.long)
    for i, x in enumerate(lists):
        x = x[:maxlen]
        if x:
            t[i, :len(x)] = torch.tensor(x)
    return t


class Practice:
    def __init__(self, worlds=32, seed=0, brain=None, vocabulary=None, lr=3e-4, rollout=20, reader=None, book=None):
        self.z = random.Random(seed)
        self.lr = lr
        import os
        self.book = reading.Book(book, random.Random(seed)) if book and os.path.exists(book) else None
        self.with_reader = False
        self.new_reader_flag = False
        self.reader_id = None
        self.words_read = 0
        if reader:                                    # hearing and speaking from the reading brain
            d = legacy.load(reader)
            vocabulary = Vocabulary(d["capacity"], d["words"])
            brain = Brain(vocab=vocabulary.capacity)
            brain.load_state_dict({k: v for k, v in d["brain"].items() if k.startswith(("word.", "hearing.", "read_out."))},
                                   strict=False)
            brain.from_reader()
            self.with_reader = self.new_reader_flag = True
            self.reader_id = d.get("reader_id")
            self.words_read = d.get("words_read", 0)
        self.n = worlds
        self.worlds = [World(random.Random(seed * 1000 + i), rest_days=True) for i in range(worlds)]
        self.teacher = [L.Teacher(random.Random(seed * 7777 + i), name="kogli" if i % 2 else "nikolai")
                       for i in range(worlds)]
        self.partner = [i + 1 if i % 2 == 0 else i - 1 for i in range(worlds)] if worlds % 2 == 0 else None
        self._pair()
        self.ws = vocabulary or Vocabulary()
        self.brain = brain or Brain(vocab=self.ws.capacity)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=lr)
        self.T = rollout
        self.h = torch.zeros(worlds, self.brain.h)
        self.last_act = self.brain.no_act(worlds)
        self.heard = [[] for _ in range(worlds)]
        self.repeat_targets = [[] for _ in range(worlds)]
        self.answers = [[] for _ in range(worlds)]
        self.perceptions = []                       # (word, properties, mask) – what it just found out
        self.things_buffer = []
        self.experience = {}                           # word → {sense: how often}
        self.senses = torch.tensor([w.senses() for w in self.worlds])
        self.steps = 0
        self.stats = {"said": 0, "reward": 0.0, "curiosity": 0.0, "instructions": 0, "fulfilled": 0, "eaten": 0, "hunger": 0.0,
                          "speech_error": 0.0, "world_error": 0.0, "repeat_error": 0.0, "hindsight_error": 0.0, "reading_error": 0.0, "answer_error": 0.0, "imagination_error": 0.0, "n": 0}
        self.curiosity_scale = 1.0

    # ------------------------------------------------------------ one practice section
    def act_once(self, h, known):
        """Decide (without learning): action, whether it also talks, what it says, and how likely that was."""
        g = self.brain
        verb, od, vd = g.decide(h)
        obj = od.sample()
        md = g.choose(h, verb)
        mod = md.sample()
        rel = relevant(verb, obj)
        mod = torch.where(rel, mod, torch.zeros_like(mod))       # "which one" only where it matters
        sd = g.switch(h)
        talks = sd.sample().bool()
        words, speak_lp, _ = g.speak(h, known)
        logp = vd.log_prob(verb) + od.log_prob(obj) + md.log_prob(mod) * rel + sd.log_prob(talks.float()) + \
            torch.where(talks, speak_lp, torch.zeros_like(speak_lp))
        return verb, obj, mod, talks, words, logp

    def _reevaluate(self, h, s):
        g = self.brain
        _, od, vd = g.decide(h, s["verb"])
        md = g.choose(h, s["verb"])
        rel = relevant(s["verb"], s["obj"])
        sd = g.switch(h)
        _, speak_lp, speak_ent = g.speak(h, s["known"], s["words"])
        talks = s["talks"]
        logp = vd.log_prob(s["verb"]) + od.log_prob(s["obj"]) + md.log_prob(s["mod"]) * rel + \
            sd.log_prob(talks.float()) + torch.where(talks, speak_lp, torch.zeros_like(speak_lp))
        ent = vd.entropy() + od.entropy() + 0.3 * md.entropy() * rel + sd.entropy() + \
            0.3 * torch.where(talks, speak_ent, torch.zeros_like(speak_ent))
        return logp, ent

    def _pair(self):
        """NikolAI (even numbers) and Kogli (odd numbers) share one world each."""
        if not self.partner:
            return
        for i in range(0, self.n, 2):
            a, b = self.worlds[i], self.worlds[i + 1]
            if b.u is not a.u:
                b.share_with(a)
            self.teacher[i].friend = (b, "kogli")
            self.teacher[i + 1].friend = (a, "nikolai")

    def section(self):
        g = self.brain
        h0 = self.h.detach()
        storage = []
        h = h0
        with torch.no_grad():
            for t in range(self.T):
                sentences = sentence_tensor(self.heard)
                h = g.think(self.senses, sentences, self.last_act, h)
                known = len(self.ws.vocab_list)
                verb, obj, mod, talks, speak_words, logp_alt = self.act_once(h, known)
                value = g.value(h).squeeze(1)
                act = g.act_index(verb, obj)
                expected = g.expectation(h, act)
                reward, newly_heard, descr, rep, hs, oks, ans, lesson = [], [], [], [], [], [], [], []
                body, evs = [], []
                for i, w in enumerate(self.worlds):          # first everyone acts …
                    k, ev = w.step(int(verb[i]), int(obj[i]), int(mod[i]))
                    if talks[i]:
                        ev["words"] = words_from(self.ws, speak_words[i])
                        self.stats["said"] += 1
                    body.append(k)
                    evs.append(ev)
                for i, (w, le) in enumerate(zip(self.worlds, self.teacher)):   # … then they hear and learn
                    ev, k = evs[i], body[i]
                    j = self.partner[i] if self.partner else None
                    le.friend_said = evs[j].get("words", []) if j is not None else []
                    sentence, l_reward, b, _ = le.respond(w, ev)
                    ans.append(self.ws.encode_ids(L.words(L.answer(le.question, w, ev, le.name))) + [END]
                               if le.question else [])
                    oks.append(bool(ev["ok"]) and ev["verb"] != "wait")
                    self._perceive(w, ev)
                    relabel = L.hindsight(ev)
                    lesson.append(L.size_lesson(ev, w))
                    hs.append(self.z.choice(relabel) if relabel else None)
                    team = TEAM * max(0.0, body[j]) if j is not None else 0.0     # shares the joy, not the pain
                    reward.append(k + team + l_reward - (SPEECH_COST if talks[i] else 0.0))
                    if ev["verb"] == "eat" and ev["ok"]:
                        self.stats["eaten"] += 1
                    self.stats["fulfilled"] += l_reward >= 1.0
                    from_friend = self.ws.encode_ids(le.friend_said) if le.friend_said else []
                    newly_heard.append(from_friend + self.ws.encode_ids(L.words(sentence)))
                    descr.append(self.ws.encode_ids(L.words(b)) + [END] if b else [])
                    prompted = L.prompted_words(sentence)
                    rep.append(self.ws.encode_ids(prompted) + [END] if prompted else [])
                    if w.steps >= 2000:          # a "day" is over: start fresh (what it learned stays in the network)
                        w.fresh()
                self._pair()
                senses_new = torch.tensor([w.senses() for w in self.worlds])
                error = ((expected - senses_new) ** 2).mean(1)
                curiosity = (error / self.curiosity_scale).clamp(max=3.0) * CURIOSITY
                self.curiosity_scale = 0.99 * self.curiosity_scale + 0.01 * float(error.mean() + 1e-6)
                r = torch.tensor(reward) + curiosity
                storage.append({"senses": self.senses, "sentences": sentences, "last": self.last_act, "verb": verb,
                                 "obj": obj, "mod": mod, "act": act, "r": r, "senses_new": senses_new, "descr": descr,
                                 "words": speak_words, "talks": talks, "known": known, "logp": logp_alt,
                                 "value": value, "repeat": self.repeat_targets, "hs": hs, "ok": oks, "lesson": lesson,
                                 "answers": self.answers})
                self.stats["reward"] += float(torch.tensor(reward).mean())
                self.stats["curiosity"] += float(curiosity.mean())
                self.stats["hunger"] += sum(w.hunger for w in self.worlds) / self.n
                self.stats["n"] += 1
                self.senses, self.last_act, self.heard, self.repeat_targets = senses_new, act, newly_heard, rep
                self.answers = ans
            h_end = g.think(self.senses, sentence_tensor(self.heard), self.last_act, h)
            value_end = g.value(h_end).squeeze(1)
        # ---------- advantages (GAE) from the values while living it
        vals_old = torch.stack([s["value"] for s in storage])
        r = torch.stack([s["r"] for s in storage])
        advantage = torch.zeros_like(r)
        gae = torch.zeros(self.n)
        next_ = value_end
        for t in reversed(range(self.T)):
            delta = r[t] + GAMMA * next_ - vals_old[t]
            gae = delta + GAMMA * LAMBDA * gae
            advantage[t] = gae
            next_ = vals_old[t]
        target = advantage + vals_old
        adv = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        logp_alt = torch.stack([s["logp"] for s in storage])
        # ---------- learning (PPO): several times over the same experience, but never too far from the old behaviour
        for epoch in range(EPOCHS):
            h = h0
            logps, ents, vals, world_v, speech_v, rep_v, answer_v = [], [], [], [], [], [], []
            h_before = []
            for s in storage:
                h_before.append(h)
                h = g.think(s["senses"], s["sentences"], s["last"], h)
                lp, en = self._reevaluate(h, s)
                logps.append(lp)
                ents.append(en)
                vals.append(g.value(h).squeeze(1))
                if epoch == 0:
                    world_v.append(F.mse_loss(g.expectation(h, s["act"]), s["senses_new"]))
                    with_s = [i for i, b in enumerate(s["descr"]) if b]
                    if with_s:
                        z = g.speech_state(h[with_s], s["senses_new"][with_s], s["act"][with_s])
                        speech_v.append(g.speech_loss(z, sentence_tensor([s["descr"][i] for i in with_s])))
                    fr = [i for i, x in enumerate(s["answers"]) if x]
                    if fr:                                    # answering: the true answer, from what it just heard
                        z = g.speech_state(h[fr], s["senses"][fr], g.no_act(len(fr)))
                        answer_v.append(g.speech_loss(z, sentence_tensor([s["answers"][i] for i in fr])))
                    prompted = [i for i, x in enumerate(s["repeat"]) if x]
                    if prompted:                                   # repeating: say what was prompted
                        target_w = torch.zeros(len(prompted), SPEECH_MAX, dtype=torch.long)
                        for j, i in enumerate(prompted):
                            x = s["repeat"][i][:SPEECH_MAX]
                            target_w[j, :len(x)] = torch.tensor(x)
                        _, lp_n, _ = g.speak(h[prompted], s["known"], target_w)
                        on = g.talk_switch(h[prompted]).squeeze(1)
                        rep_v.append(-lp_n.mean() + F.binary_cross_entropy_with_logits(on, torch.ones_like(on)))
            lp_t = torch.stack(logps)
            ratio = torch.exp(lp_t - logp_alt)
            pol = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()
            value_v = F.mse_loss(torch.stack(vals), target)
            ent = torch.stack(ents).mean()
            loss = pol + 0.5 * value_v - 0.01 * ent
            if epoch == 0:
                world_l = torch.stack(world_v).mean()
                speech_l = torch.stack(speech_v).mean() if speech_v else torch.tensor(0.0)
                rep_l = torch.stack(rep_v).mean() if rep_v else torch.tensor(0.0)
                hind_l = self._hindsight(storage, h_before)
                size_l = self._size_lessons(storage, h_before)
                answer_l = torch.stack(answer_v).mean() if answer_v else torch.tensor(0.0)
                loss = loss + world_l + 0.5 * speech_l + 0.3 * rep_l + HINDSIGHT_WEIGHT * hind_l + 1.0 * answer_l + \
                    SIZE_WEIGHT * size_l
                read_l = torch.tensor(0.0)
                if self.book:                         # a few lines of its book
                    x = reading.batch_of(self.book, self.ws, 16, 64)
                    read_l = g.reading_loss(x)
                    self.words_read += x.numel()
                    loss = loss + READING_WEIGHT * read_l
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 0.5)
            self.opt.step()
        self.h = h.detach()
        self.steps += self.T * self.n
        self.stats["speech_error"] += float(speech_l.detach())
        self.stats["world_error"] += float(world_l.detach())
        self.stats["repeat_error"] += float(rep_l.detach())
        self.stats["hindsight_error"] += float(hind_l.detach())
        self.stats["reading_error"] += float(read_l.detach())
        self.stats["answer_error"] += float(answer_l.detach())
        self._practise_imagination()
        return float(loss.detach())

    def _perceive(self, w, ev):
        """What it finds out about things in this step: seeing, taking, carrying, eating."""
        if len(self.perceptions) >= PERCEPTIONS_MAX:
            return
        experienced = []
        if ev["verb"] == "take" and ev["ok"] and ev["obj"] in V.THING_WORD:
            experienced.append((ev["obj"], "taking", ev.get("size")))
        if ev["verb"] == "eat" and ev["ok"]:
            experienced.append(("apple", "eating", ev.get("size")))
        if ev["verb"] == "build" and ev["ok"]:
            experienced += [("stone_axe", "seeing", None), ("stone_axe", "carrying", None)]
        visible_now = V.visible_things(w)
        if visible_now and self.z.random() < 0.3:
            d, size = self.z.choice(visible_now)
            experienced.append((d, "seeing", size))
        carries = [d for d, n in w.backpack.items() if n > 0]
        if carries and self.z.random() < 0.1:
            d = self.z.choice(carries)
            experienced.append((d, "carrying", V.size_in(w.backpack, w.backpack_big, d, self.z)))
        for thing, sense, size in experienced:
            x, m = V.perceive(thing, sense, self.z, size)
            word = V.THING_WORD[thing]
            # the words it hears for it: "stone" – or, as the teacher would say it, "big apple" (never a held-out
            # combination: those it only hears in the test)
            if size and thing in V.SIZED_THINGS and not L.held_out(thing, size) and self.z.random() < 0.5:
                word = f"{size} {word}"
            self.perceptions.append((self.ws.encode_ids(L.words(word)), x, m))
            e = self.experience.setdefault(word, {})
            e[sense] = e.get(sense, 0) + 1

    def _practise_imagination(self):
        """Expectation against perception (Gaussian: mean and uncertainty), several times per section, from a memory of
        recent experiences. Only the imagination part learns here – how it hears stays as it is.
        Words for things it never touched are kept in the fog."""
        g = self.brain
        self.things_buffer = (self.things_buffer + self.perceptions)[-IMAGINATION_BUFFER:]
        self.perceptions = []
        if len(self.things_buffer) < 32:
            return
        if not hasattr(self, "imag_opt") or self.imag_opt_for is not g:
            self.imag_opt = torch.optim.Adam(g.imagination.parameters(), lr=2e-3)
            self.imag_opt_for = g
        unknown = [[self.ws.index[w]] for w in V.UNKNOWN if w in self.ws.index]
        for _ in range(IMAGINATION_STEPS):
            sample_ = self.z.sample(self.things_buffer, min(128, len(self.things_buffer)))
            words, x, m = zip(*sample_)
            with torch.no_grad():
                c = g.hear(sentence_tensor(list(words)))
            outp = g.imagination(c)
            mu, lv = torch.sigmoid(outp[:, :V.P]), outp[:, V.P:].clamp(-7, 1)
            x, m = torch.stack(x), torch.stack(m)
            loss = (0.5 * (lv + (x - mu) ** 2 / lv.exp()) * m).sum() / m.sum()
            if unknown:                               # named, but never touched: stay unsure
                with torch.no_grad():
                    cu = g.hear(sentence_tensor(self.z.sample(unknown, min(8, len(unknown)))))
                au = g.imagination(cu)
                loss = loss + 0.3 * (((torch.sigmoid(au[:, :V.P]) - 0.5) ** 2).mean()
                                           + ((au[:, V.P:] - math.log(0.25)) ** 2).mean())
            self.imag_opt.zero_grad()
            loss.backward()
            self.imag_opt.step()
        self.stats["imagination_error"] += float(((x - mu).abs() * m).sum().detach() / m.sum())

    def _size_lessons(self, storage, h_before):
        """Version 17 – lessons for the hands: whenever it took, put down or ate a big or a small one while both
        were there, it replays that moment as if it had heard "Take a big apple!" (the size it really chose) and
        learns: whoever hears this chooses that one. Never for the held-out combinations."""
        g = self.brain
        items = [(t, i, sentence) for t, s in enumerate(storage) for i, sentence in enumerate(s["lesson"]) if sentence]
        if not items:
            return torch.tensor(0.0)
        items = self.z.sample(items, min(64, len(items)))
        h = torch.stack([h_before[t][i] for t, i, _ in items]).detach()
        senses = torch.stack([storage[t]["senses"][i] for t, i, _ in items])
        last = torch.stack([storage[t]["last"][i] for t, i, _ in items])
        verb = torch.stack([storage[t]["verb"][i] for t, i, _ in items])
        sentences = [self.ws.encode_ids(L.words(x)) for _, _, x in items]
        h = g.think(senses, sentence_tensor(sentences), last, h)
        target = torch.tensor([L.MOD_OF[x] for _, _, x in items])
        return -g.choose(h, verb).log_prob(target).mean()

    def _hindsight(self, storage, h_before):
        """One successful action per world: the steps before it once more, with the fitting sentence at the start."""
        g = self.brain
        samples = []
        for i in range(self.n):
            hits = [t for t, s in enumerate(storage) if s["hs"][i]]
            sized = [t for t in hits if storage[t]["hs"][i] in L.MOD_OF]
            if hits:
                # version 17: half of the time, if there is one, a lesson with a size ("Take a big apple!") – rarer,
                # but it is the only way to learn which one the hands should choose
                e = self.z.choice(sized if sized and self.z.random() < 0.5 else hits)
                samples.append((i, max(0, e - self.z.randint(0, HINDSIGHT_MAX - 1)), e, storage[e]["hs"][i]))
        if not samples:
            return torch.tensor(0.0)
        h = torch.stack([h_before[s0][i] for i, s0, _, _ in samples]).detach()
        total_sum, n_total = torch.tensor(0.0), 0
        for k in range(max(e - s0 for _, s0, e, _ in samples) + 1):
            ts = [min(s0 + k, e) for _, s0, e, _ in samples]
            # only steps that worked are imitated (failed attempts on the way are not "how to do it")
            active = torch.tensor([s0 + k <= e and storage[min(s0 + k, e)]["ok"][i] for i, s0, e, _ in samples],
                                 dtype=torch.float)
            sentences = []
            for (i, s0, e, sentence), t in zip(samples, ts):
                heard = [int(x) for x in storage[t]["sentences"][i] if int(x)] if storage[t]["sentences"].shape[1] else []
                sentences.append(self.ws.encode_ids(L.words(sentence)) + heard if k == 0 else heard)
            senses = torch.stack([storage[t]["senses"][i] for (i, _, _, _), t in zip(samples, ts)])
            last = torch.stack([storage[t]["last"][i] for (i, _, _, _), t in zip(samples, ts)])
            verb = torch.stack([storage[t]["verb"][i] for (i, _, _, _), t in zip(samples, ts)])
            obj = torch.stack([storage[t]["obj"][i] for (i, _, _, _), t in zip(samples, ts)])
            # which one (version 17): at the action itself, what the sentence asks for ("a big stone" → big,
            # "a stone" → any); before it, what it really chose
            mod = torch.tensor([L.MOD_OF.get(sentence, 0) if s0 + k >= e else int(storage[t]["mod"][i])
                                for (i, s0, e, sentence), t in zip(samples, ts)])
            h = g.think(senses, sentence_tensor(sentences), last, h)
            _, od, vd = g.decide(h, verb)
            lp = vd.log_prob(verb) + od.log_prob(obj) + g.choose(h, verb).log_prob(mod) * relevant(verb, obj)
            total_sum = total_sum - (lp * active).sum()
            n_total += int(active.sum())
        return total_sum / max(1, n_total)

    # ------------------------------------------------------------ saving
    def save(self, fpath, extra=None):
        torch.save({"brain": self.brain.state_dict(), "opt": self.opt.state_dict(), "words": self.ws.vocab_list,
                    "capacity": self.ws.capacity, "words_read": self.words_read, "with_reader": self.with_reader,
                    "reader_id": self.reader_id, "experience": self.experience, "things_buffer": self.things_buffer[-1000:],
                    "steps": self.steps, "extra": extra or {}}, fpath + ".tmp")
        import os
        os.replace(fpath + ".tmp", fpath)

    def restore(self, fpath):
        d = legacy.load(fpath)
        if self.new_reader_flag:          # a new reading brain moves in: keep body, senses, actions; language from reading
            old = {k: v for k, v in d["brain"].items() if not k.startswith(LANGUAGE_PARTS)}
            self.brain.adopt(old)
            self.opt = torch.optim.Adam(self.brain.parameters(), lr=self.lr)
            self.steps = d["steps"]
            return d.get("extra", {})
        if d.get("with_reader"):                                  # a brain that already reads: take it as it is
            self.ws = Vocabulary(d["capacity"], d["words"])
            self.brain = Brain(vocab=self.ws.capacity)
            missing = self.brain.load_grown(d["brain"])
            if missing:
                print("new parts in the brain (start fresh):", missing, flush=True)
            self.opt = torch.optim.Adam(self.brain.parameters(), lr=self.lr)
            try:
                self.opt.load_state_dict(d["opt"])
            except (ValueError, KeyError):
                pass
            self.steps, self.words_read, self.with_reader = d["steps"], d.get("words_read", 0), True
            self.reader_id = d.get("reader_id")
            self.experience = d.get("experience", {})
            self.things_buffer = d.get("things_buffer", [])
            return d.get("extra", {})
        if d.get("extra", {}).get("version", 1) >= 5:          # from version 5 on NikolAI learns English
            self.ws = Vocabulary(self.ws.capacity, d["words"])
        # an older (German) brain keeps what it knows about the world, but starts with an empty vocabulary
        newly_built = self.brain.adopt(d["brain"])
        if not newly_built:
            try:
                self.opt.load_state_dict(d["opt"])
            except (ValueError, KeyError):
                pass                                   # new parts in the brain: the optimizer starts fresh
        self.steps = d["steps"]
        return d.get("extra", {})

    def report(self):
        s = self.stats
        n = max(1, s["n"])
        out = {"steps": self.steps, "reward": s["reward"] / n, "curiosity": s["curiosity"] / n,
               "hunger": s["hunger"] / n, "eaten_per_1000": 1000 * s["eaten"] / (n * self.n),
               "instructions_done_per_1000": 1000 * s["fulfilled"] / (n * self.n),
               "speech_error": s["speech_error"] / max(1, n / self.T),
               "world_error": s["world_error"] / max(1, n / self.T), "words": len(self.ws.vocab_list) - 3,
               "said_per_1000": 1000 * s["said"] / (n * self.n),
               "repeat_error": s["repeat_error"] / max(1, n / self.T),
               "hindsight_error": s["hindsight_error"] / max(1, n / self.T),
               "reading_error": s["reading_error"] / max(1, n / self.T), "words_read": self.words_read,
               "answer_error": s["answer_error"] / max(1, n / self.T),
               "imagination_error": s["imagination_error"] / max(1, n / self.T),
               "where_asked_per_1000": 1000 * sum(le.requests["where_q"] - le._w0 for le in self.teacher) / (n * self.n),
               "given_per_1000": 1000 * sum(le.requests["given"] - le._b0 for le in self.teacher) / (n * self.n)}
        for le in self.teacher:
            le._b0 = le.requests["given"]
            le._w0 = le.requests["where_q"]
        for k in s:
            s[k] = 0 if isinstance(s[k], int) else 0.0
        return out


def run(minutes=5, worlds=32, seed=0, report_all=2000):
    torch.manual_seed(seed)
    u = Practice(worlds=worlds, seed=seed)
    t0, i = time.time(), 0
    while time.time() - t0 < minutes * 60:
        u.section()
        i += 1
        if i % (report_all // u.T) == 0:
            b = u.report()
            print(f"{int(time.time() - t0):5d}s {b['steps']:9d} steps | hunger {b['hunger']:.2f} | "
                  f"apples/1000 {b['eaten_per_1000']:.1f} | instructions/1000 {b['instructions_done_per_1000']:.1f} | "
                  f"speech error {b['speech_error']:.2f} | world error {b['world_error']:.4f}", flush=True)
    return u
