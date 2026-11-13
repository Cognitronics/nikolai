"""Practice: the brain lives in several copies of the world at once (like many hours of practice at the same time) and learns.

Learning signals:
  * Body: hunger hurts, getting full feels good.
  * Curiosity: what the world model didn't predict is exciting (reward = surprise, Pathak et al. 2017).
    The better the world model gets, the more boring the familiar becomes – the agent looks for something new.
  * Teacher: praise when an instruction is done.
  * Speaking: the brain tries to predict what the teacher is about to say about its action.
  * Hindsight (version 6): after every successful action, practice asks: which instruction would this have fulfilled?
    ("You took a stone" → "Take a stone!"). Then it replays the last few steps as if it had heard that sentence at the
    start, and learns: whoever hears this sentence does this. So every action becomes a lesson
    (Andrychowicz et al. 2017, "Hindsight Experience Replay"; Ghosh et al. 2019, GCSL; Chan et al. 2019).
  * Answering (version 8): if the teacher asks "Where are you?", speaking learns the true answer in the next step
    ("i am in the forest") – from what it just heard and how it is doing.
  * Friends (version 10): the worlds are shared in pairs – NikolAI and Kogli (one brain, two bodies, like twins).
    Each hears what the other says and feels it too when the other gets full (a shared reward, as with cooperating
    agents in research, e.g. Foerster et al. 2016, Lowe et al. 2017).
  * Reading (version 7): alongside, it keeps reading its book (guessing the next word). So hearing doesn't forget
    what it knows from books while it learns what the words refer to in its world.
Method: PPO (actor-critic with GAE); the world model and speaking are trained at the same time.
"""

import random
import time

import torch
import torch.nn.functional as F

from . import compat
from . import teacher as L
from . import reading
from .brain import END, PAD, TALK_MAX, Brain, Vocabulary
from .world import VERBS, World

SAY = VERBS.index("say")


def words_from(ws, row):
    return [ws.word(int(i)) for i in row if int(i) not in (PAD, END)]

GAMMA, LAMBDA = 0.97, 0.95
CURIOSITY = 0.05
EPOCHS, CLIP = 3, 0.2           # PPO: three passes over the same experience, change per step limited
HINDSIGHT_MAX = 5               # the sentence is inserted "after the fact" up to this many steps before the action
HINDSIGHT_WEIGHT = 0.3
READING_WEIGHT = 0.5
TEAM = 0.5                      # how much it shares the joy when the friend gets full
LANGUAGE_PARTS = ("word.", "hearing.", "read_out.", "hear_proj.", "speaking.", "speak_out.", "speak_start.",
                  "talk_cell.", "talk_out.")
TALK_COST = 0.04              # talking costs a little effort – otherwise it babbles nonstop


def sentence_tensor(seqs, maxlen=24):
    n = max([len(x) for x in seqs] + [0])
    t = torch.zeros(len(seqs), min(n, maxlen), dtype=torch.long)
    for i, x in enumerate(seqs):
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
        self.fresh_reader = False
        self.reader_id = None
        self.words_read = 0
        if reader:                                    # hearing and speaking from the reading brain
            d = compat.migrate(torch.load(reader, weights_only=False))
            vocabulary = Vocabulary(d["size"], d["vocab_list"])
            brain = Brain(vocab=vocabulary.size)
            brain.load_state_dict({k: v for k, v in d["brain"].items() if k.startswith(("word.", "hearing.", "read_out."))},
                                   strict=False)
            brain.from_reader()
            self.with_reader = self.fresh_reader = True
            self.reader_id = d.get("reader_id")
            self.words_read = d.get("words_read", 0)
        self.n = worlds
        self.worlds = [World(random.Random(seed * 1000 + i), rest_days=True) for i in range(worlds)]
        self.teacher = [L.Teacher(random.Random(seed * 7777 + i), name="kogli" if i % 2 else "nikolai")
                       for i in range(worlds)]
        self.partner = [i + 1 if i % 2 == 0 else i - 1 for i in range(worlds)] if worlds % 2 == 0 else None
        self._pair_up()
        self.ws = vocabulary or Vocabulary()
        self.brain = brain or Brain(vocab=self.ws.size)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=lr)
        self.T = rollout
        self.h = torch.zeros(worlds, self.brain.h)
        self.last_action = self.brain.no_action(worlds)
        self.heard = [[] for _ in range(worlds)]
        self.repeating = [[] for _ in range(worlds)]
        self.answers = [[] for _ in range(worlds)]
        self.senses = torch.tensor([w.senses() for w in self.worlds])
        self.steps = 0
        self.stats = {"said": 0, "rewards": 0.0, "curiosity": 0.0, "instructions": 0, "fulfilled": 0, "eaten": 0, "hunger": 0.0,
                          "describe_error": 0.0, "world_error": 0.0, "repeat_error": 0.0, "hindsight_error": 0.0, "read_error": 0.0, "answer_error": 0.0, "n": 0}
        self.curiosity_scale = 1.0

    # ------------------------------------------------------------ one round of practice
    def choose_action(self, h, known):
        """Deciding (without learning): the action, whether it talks alongside, what it says, and how likely that was."""
        g = self.brain
        verb, od, vd = g.decide(h)
        obj = od.sample()
        sd = g.talk_switch(h)
        talks = sd.sample().bool()
        words, talk_lp, _ = g.talk(h, known)
        logp = vd.log_prob(verb) + od.log_prob(obj) + sd.log_prob(talks.float()) + torch.where(
            talks, talk_lp, torch.zeros_like(talk_lp))
        return verb, obj, talks, words, logp

    def _reevaluate(self, h, s):
        g = self.brain
        _, od, vd = g.decide(h, s["verb"])
        sd = g.talk_switch(h)
        _, talk_lp, talk_ent = g.talk(h, s["known"], s["words"])
        talks = s["talks"]
        logp = vd.log_prob(s["verb"]) + od.log_prob(s["obj"]) + sd.log_prob(talks.float()) + torch.where(
            talks, talk_lp, torch.zeros_like(talk_lp))
        ent = vd.entropy() + od.entropy() + sd.entropy() + 0.3 * torch.where(talks, talk_ent, torch.zeros_like(talk_ent))
        return logp, ent

    def _pair_up(self):
        """NikolAI (even numbers) and Kogli (odd numbers) share one world each."""
        if not self.partner:
            return
        for i in range(0, self.n, 2):
            a, b = self.worlds[i], self.worlds[i + 1]
            if b.u is not a.u:
                b.share_with(a)
            self.teacher[i].friend = (b, "kogli")
            self.teacher[i + 1].friend = (a, "nikolai")

    def practice_round(self):
        g = self.brain
        h0 = self.h.detach()
        buffer = []
        h = h0
        with torch.no_grad():
            for t in range(self.T):
                sentences = sentence_tensor(self.heard)
                h = g.tick(self.senses, sentences, self.last_action, h)
                known = len(self.ws.words_list)
                verb, obj, talks, talk_words, logp_old = self.choose_action(h, known)
                value = g.value(h).squeeze(1)
                act = g.act_id(verb, obj)
                predicted = g.predict(h, act)
                reward, new_heard, descr, repeat_targets, hs, oks, answers_now = [], [], [], [], [], [], []
                body, evs = [], []
                for i, w in enumerate(self.worlds):          # first everyone acts …
                    k, ev = w.step(int(verb[i]), int(obj[i]))
                    if talks[i]:
                        ev["words"] = words_from(self.ws, talk_words[i])
                        self.stats["said"] += 1
                    body.append(k)
                    evs.append(ev)
                for i, (w, le) in enumerate(zip(self.worlds, self.teacher)):   # … then they hear and learn
                    ev, k = evs[i], body[i]
                    j = self.partner[i] if self.partner else None
                    le.friend_said = evs[j].get("words", []) if j is not None else []
                    sentence, t_reward, b, _ = le.respond(w, ev)
                    answers_now.append(self.ws.encode_words(L.words(L.answer(le.question, w, ev, le.name))) + [END]
                               if le.question else [])
                    oks.append(bool(ev["ok"]) and ev["verb"] != "wait")
                    relabels = L.hindsight(ev)
                    hs.append(self.z.choice(relabels) if relabels else None)
                    team = TEAM * max(0.0, body[j]) if j is not None else 0.0     # shares the joy, not the pain
                    reward.append(k + team + t_reward - (TALK_COST if talks[i] else 0.0))
                    if ev["verb"] == "eat" and ev["ok"]:
                        self.stats["eaten"] += 1
                    self.stats["fulfilled"] += t_reward >= 1.0
                    from_friend = self.ws.encode_words(le.friend_said) if le.friend_said else []
                    new_heard.append(from_friend + self.ws.encode_words(L.words(sentence)))
                    descr.append(self.ws.encode_words(L.words(b)) + [END] if b else [])
                    modelled = L.modelled_words(sentence)
                    repeat_targets.append(self.ws.encode_words(modelled) + [END] if modelled else [])
                    if w.steps >= 2000:          # a "day" is over: start fresh (memory stays in the network)
                        w.reset()
                self._pair_up()
                senses_new = torch.tensor([w.senses() for w in self.worlds])
                error = ((predicted - senses_new) ** 2).mean(1)
                curiosity = (error / self.curiosity_scale).clamp(max=3.0) * CURIOSITY
                self.curiosity_scale = 0.99 * self.curiosity_scale + 0.01 * float(error.mean() + 1e-6)
                r = torch.tensor(reward) + curiosity
                buffer.append({"senses": self.senses, "sentences": sentences, "last": self.last_action, "verb": verb,
                                 "obj": obj, "act": act, "r": r, "senses_new": senses_new, "descr": descr,
                                 "words": talk_words, "talks": talks, "known": known, "logp": logp_old,
                                 "value": value, "repeating": self.repeating, "hs": hs, "ok": oks,
                                 "answers": self.answers})
                self.stats["rewards"] += float(torch.tensor(reward).mean())
                self.stats["curiosity"] += float(curiosity.mean())
                self.stats["hunger"] += sum(w.hunger for w in self.worlds) / self.n
                self.stats["n"] += 1
                self.senses, self.last_action, self.heard, self.repeating = senses_new, act, new_heard, repeat_targets
                self.answers = answers_now
            h_end = g.tick(self.senses, sentence_tensor(self.heard), self.last_action, h)
            value_end = g.value(h_end).squeeze(1)
        # ---------- advantages (GAE) from the values seen while acting
        values_old = torch.stack([s["value"] for s in buffer])
        r = torch.stack([s["r"] for s in buffer])
        advantage = torch.zeros_like(r)
        gae = torch.zeros(self.n)
        next_value = value_end
        for t in reversed(range(self.T)):
            delta = r[t] + GAMMA * next_value - values_old[t]
            gae = delta + GAMMA * LAMBDA * gae
            advantage[t] = gae
            next_value = values_old[t]
        goal = advantage + values_old
        adv = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        logp_old = torch.stack([s["logp"] for s in buffer])
        # ---------- learning (PPO): several passes over the same experience, but never too far from the old behaviour
        for epoch in range(EPOCHS):
            h = h0
            logps, ents, value_list, world_v, speak_v, repeat_v, answer_v = [], [], [], [], [], [], []
            h_before = []
            for s in buffer:
                h_before.append(h)
                h = g.tick(s["senses"], s["sentences"], s["last"], h)
                lp, en = self._reevaluate(h, s)
                logps.append(lp)
                ents.append(en)
                value_list.append(g.value(h).squeeze(1))
                if epoch == 0:
                    world_v.append(F.mse_loss(g.predict(h, s["act"]), s["senses_new"]))
                    with_cue = [i for i, b in enumerate(s["descr"]) if b]
                    if with_cue:
                        z = g.speak_state(h[with_cue], s["senses_new"][with_cue], s["act"][with_cue])
                        speak_v.append(g.speak_loss(z, sentence_tensor([s["descr"][i] for i in with_cue])))
                    asked_i = [i for i, x in enumerate(s["answers"]) if x]
                    if asked_i:                                    # answering: the true answer, from what it just heard
                        z = g.speak_state(h[asked_i], s["senses"][asked_i], g.no_action(len(asked_i)))
                        answer_v.append(g.speak_loss(z, sentence_tensor([s["answers"][i] for i in asked_i])))
                    modelled = [i for i, x in enumerate(s["repeating"]) if x]
                    if modelled:                                   # repeating: say what was modelled
                        goal_w = torch.zeros(len(modelled), TALK_MAX, dtype=torch.long)
                        for j, i in enumerate(modelled):
                            x = s["repeating"][i][:TALK_MAX]
                            goal_w[j, :len(x)] = torch.tensor(x)
                        _, lp_n, _ = g.talk(h[modelled], s["known"], goal_w)
                        switch_logit = g.speak_switch(h[modelled]).squeeze(1)
                        repeat_v.append(-lp_n.mean() + F.binary_cross_entropy_with_logits(switch_logit, torch.ones_like(switch_logit)))
            lp_t = torch.stack(logps)
            ratio = torch.exp(lp_t - logp_old)
            pol = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()
            value_v = F.mse_loss(torch.stack(value_list), goal)
            ent = torch.stack(ents).mean()
            loss = pol + 0.5 * value_v - 0.01 * ent
            if epoch == 0:
                world_l = torch.stack(world_v).mean()
                speak_l = torch.stack(speak_v).mean() if speak_v else torch.tensor(0.0)
                repeat_l = torch.stack(repeat_v).mean() if repeat_v else torch.tensor(0.0)
                hind_l = self._hindsight(buffer, h_before)
                answer_l = torch.stack(answer_v).mean() if answer_v else torch.tensor(0.0)
                loss = loss + world_l + 0.5 * speak_l + 0.3 * repeat_l + HINDSIGHT_WEIGHT * hind_l + 1.0 * answer_l
                read_l = torch.tensor(0.0)
                if self.book:                         # a few lines of its book
                    x = reading.batch(self.book, self.ws, 16, 64)
                    read_l = g.reading_loss(x)
                    self.words_read += x.numel()
                    loss = loss + READING_WEIGHT * read_l
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 0.5)
            self.opt.step()
        self.h = h.detach()
        self.steps += self.T * self.n
        self.stats["describe_error"] += float(speak_l.detach())
        self.stats["world_error"] += float(world_l.detach())
        self.stats["repeat_error"] += float(repeat_l.detach())
        self.stats["hindsight_error"] += float(hind_l.detach())
        self.stats["read_error"] += float(read_l.detach())
        self.stats["answer_error"] += float(answer_l.detach())
        return float(loss.detach())

    def _hindsight(self, buffer, h_before):
        """One successful action per world: the steps before it once more, with the matching sentence at the start."""
        g = self.brain
        samples = []
        for i in range(self.n):
            hits = [t for t, s in enumerate(buffer) if s["hs"][i]]
            if hits:
                e = self.z.choice(hits)
                samples.append((i, max(0, e - self.z.randint(0, HINDSIGHT_MAX - 1)), e, buffer[e]["hs"][i]))
        if not samples:
            return torch.tensor(0.0)
        h = torch.stack([h_before[s0][i] for i, s0, _, _ in samples]).detach()
        total_sum, n_active = torch.tensor(0.0), 0
        for k in range(max(e - s0 for _, s0, e, _ in samples) + 1):
            ts = [min(s0 + k, e) for _, s0, e, _ in samples]
            # only steps that worked are imitated (failed attempts on the way are not "how to do it")
            active = torch.tensor([s0 + k <= e and buffer[min(s0 + k, e)]["ok"][i] for i, s0, e, _ in samples],
                                 dtype=torch.float)
            sentences = []
            for (i, s0, e, sentence), t in zip(samples, ts):
                heard = [int(x) for x in buffer[t]["sentences"][i] if int(x)] if buffer[t]["sentences"].shape[1] else []
                sentences.append(self.ws.encode_words(L.words(sentence)) + heard if k == 0 else heard)
            senses = torch.stack([buffer[t]["senses"][i] for (i, _, _, _), t in zip(samples, ts)])
            last = torch.stack([buffer[t]["last"][i] for (i, _, _, _), t in zip(samples, ts)])
            verb = torch.stack([buffer[t]["verb"][i] for (i, _, _, _), t in zip(samples, ts)])
            obj = torch.stack([buffer[t]["obj"][i] for (i, _, _, _), t in zip(samples, ts)])
            h = g.tick(senses, sentence_tensor(sentences), last, h)
            _, od, vd = g.decide(h, verb)
            lp = vd.log_prob(verb) + od.log_prob(obj)
            total_sum = total_sum - (lp * active).sum()
            n_active += int(active.sum())
        return total_sum / max(1, n_active)

    # ------------------------------------------------------------ saving
    def save_brain(self, file_path, extra=None):
        torch.save({"brain": self.brain.state_dict(), "opt": self.opt.state_dict(), "vocab_list": self.ws.words_list,
                    "size": self.ws.size, "words_read": self.words_read, "with_reader": self.with_reader,
                    "reader_id": self.reader_id,
                    "steps": self.steps, "extra": extra or {}}, file_path + ".tmp")
        import os
        os.replace(file_path + ".tmp", file_path)

    def load_from(self, file_path):
        d = compat.migrate(torch.load(file_path, weights_only=False))
        if self.fresh_reader:          # a new reading brain moves in: keep body, senses, actions; language from reading
            old = {k: v for k, v in d["brain"].items() if not k.startswith(LANGUAGE_PARTS)}
            self.brain.adopt(old)
            self.opt = torch.optim.Adam(self.brain.parameters(), lr=self.lr)
            self.steps = d["steps"]
            return d.get("extra", {})
        if d.get("with_reader"):                                  # a brain that already reads: take it as it is
            self.ws = Vocabulary(d["size"], d["vocab_list"])
            self.brain = Brain(vocab=self.ws.size)
            self.brain.load_state_dict(d["brain"])
            self.opt = torch.optim.Adam(self.brain.parameters(), lr=self.lr)
            try:
                self.opt.load_state_dict(d["opt"])
            except (ValueError, KeyError):
                pass
            self.steps, self.words_read, self.with_reader = d["steps"], d.get("words_read", 0), True
            self.reader_id = d.get("reader_id")
            return d.get("extra", {})
        if d.get("extra", {}).get("version", 1) >= 5:          # from version 5 on NikolAI learns English
            self.ws = Vocabulary(self.ws.size, d["vocab_list"])
        # an older (German) brain keeps what it knows about the world, but starts with an empty vocabulary
        rebuilt = self.brain.adopt(d["brain"])
        if not rebuilt:
            try:
                self.opt.load_state_dict(d["opt"])
            except (ValueError, KeyError):
                pass                                   # new parts in the brain: the optimizer starts fresh
        self.steps = d["steps"]
        return d.get("extra", {})

    def report(self):
        s = self.stats
        n = max(1, s["n"])
        out = {"steps": self.steps, "rewards": s["rewards"] / n, "curiosity": s["curiosity"] / n,
               "hunger": s["hunger"] / n, "eaten_per_1000": 1000 * s["eaten"] / (n * self.n),
               "instructions_done_per_1000": 1000 * s["fulfilled"] / (n * self.n),
               "describe_error": s["describe_error"] / max(1, n / self.T),
               "world_error": s["world_error"] / max(1, n / self.T), "words_known": len(self.ws.words_list) - 3,
               "said_per_1000": 1000 * s["said"] / (n * self.n),
               "repeat_error": s["repeat_error"] / max(1, n / self.T),
               "hindsight_error": s["hindsight_error"] / max(1, n / self.T),
               "read_error": s["read_error"] / max(1, n / self.T), "words_read": self.words_read,
               "answer_error": s["answer_error"] / max(1, n / self.T),
               "asked_where_per_1000": 1000 * sum(le.requests["where_is"] - le._w0 for le in self.teacher) / (n * self.n),
               "given_per_1000": 1000 * sum(le.requests["given"] - le._b0 for le in self.teacher) / (n * self.n)}
        for le in self.teacher:
            le._b0 = le.requests["given"]
            le._w0 = le.requests["where_is"]
        for k in s:
            s[k] = 0 if isinstance(s[k], int) else 0.0
        return out


def run(minutes=5, worlds=32, seed=0, report_every=2000):
    torch.manual_seed(seed)
    u = Practice(worlds=worlds, seed=seed)
    t0, i = time.time(), 0
    while time.time() - t0 < minutes * 60:
        u.practice_round()
        i += 1
        if i % (report_every // u.T) == 0:
            b = u.report()
            print(f"{int(time.time() - t0):5d}s {b['steps']:9d} steps | hunger {b['hunger']:.2f} | "
                  f"apples/1000 {b['eaten_per_1000']:.1f} | instructions/1000 {b['instructions_done_per_1000']:.1f} | "
                  f"describe error {b['describe_error']:.2f} | world error {b['world_error']:.4f}", flush=True)
    return u
