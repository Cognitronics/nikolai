"""Tests: has the brain really learned something? Every test compares against a baseline, so nothing is flattered.

Phase 1 – Feeds itself: 600 steps without a teacher. Average hunger, apples eaten. Baseline: acting at random.
Phase 2 – Understands instructions: every instruction is tested with and without the spoken sentence (same start),
          15 steps, 30 trials. "Understood" means: with the sentence it works in at least 70 % and at least 30 points
          more often than without. What it does anyway (e.g. fetching apples) is "not testable". Two instructions
          are never given in training ("Put down the apple!", "Take some bast!") – only their words in other sentences.
Phase 3 – Describes what happened: after each action NikolAI says what happened. Word for word like the teacher?
Phase 4 – Asks for things: NikolAI stands in the workshop (far from the apple tree). Does it say "apple" on its own
          when hungry – and not when full? With the instruction "Build a stone axe!": does it say "bast" when only bast
          is missing – and not when nothing is? 10 steps, baseline: acting and talking at random.
Phase 4c – First person: how often does it ask with "I want …" instead of just the word?
Phase 4d – Repeats on request ("Say …!").
Phase 5 – Uses what it is told: hungry in the hut, the apple tree is empty, a single apple lies somewhere (forest,
          river or workshop). With "There is an apple at the river." – does it go there and take it within 4 steps?
          (Two steps are enough if it understood; guessing finds it about one time in three.) Without the sentence
          and at random for comparison.
Phase 5b – Asks where: hungry, no hint – does it ask "where … apple?" on its own (and less often when full)?
Phase 6 – Reads: on text it never read, how well does it guess the next word? (perplexity – lower is better;
          baseline: only knowing how frequent each word is.) Milestone: less than a third of the baseline.
Phase 6b – Understands words it only knows from books: instructions with words the teacher never uses
          ("Grab a stone!" instead of "Take a stone!"), tested like Phase 2 (Hill et al. 2020), but from a start that
          doesn't give the answer away (several things to choose from, or the hut with four ways) and with only
          4 steps (one or two are enough if it understood).
          Control: the same sentence with a made-up word ("Blick a stone!"). Maybe it only reacts to "stone" –
          then the made-up word works just as well. It only counts if the real word works clearly better.
Phase 8 – Tells its friend where food is: NikolAI stands where an apple lies; Kogli is hungry in the hut, the tree is
          empty, no teacher. Does Kogli get the apple within 10 steps? Once with them hearing each other, once with
          the line cut (they can't hear each other). If talking helps clearly, they really communicate. We also count
          how often NikolAI names the right place.
Phase 9 – Imagines what it knows: from the heard word alone, how close are its expected looks (colour, size, shape,
          parts) to what the thing is really like? Baseline: the average of all things. And does it stay unsure about
          words it never touched (e.g. "tiger", "window" – never trained, not even as fog)?
Phase 7 – Answers questions about itself: a random situation (place, hunger, what it carries, what it just did),
          then the teacher asks ("Where are you?" …). Is its answer exactly the true one? Baseline: always giving the
          most frequent answer. A question type counts when it is right in at least 70 % and 30 points above that
          baseline; the milestone needs four of the five types (not counting "Who are you?").
"""

import random

import torch

from . import teacher as L
from .brain import END
from .practice import SAY, sentence_tensor, words_from
from .world import OBJECTS, VERBS, PATHS, World

WHERE_STEPS = 4
WHERE_PLACES = ("forest", "river", "workshop")


def _start(world, task):
    if task[0] == "put":
        world.backpack[task[1]] = 1
    if task == ("eat", "apple"):
        world.hunger = 0.6


def _start_fair(world, task):
    """Phase 6b: a start that doesn't give the answer away. Several things lie around (take the right one), it
    carries several things (put down the right one), or it stands in the hut with four ways and isn't hungry."""
    v, d = task
    world.hunger = 0.1
    if v == "take":
        world.place = "workshop"                       # nothing grows here – only what lies on the floor
        for x in ("apple", "branch", "stone", "bast"):
            world.ground["workshop"][x] = 1
    elif v == "put":
        for x in ("apple", "branch", "stone"):
            world.backpack[x] = 1
    elif v == "go":
        world.place = "hut"


FAIR_STEPS = 4           # Phase 6b: one or two steps are enough if it understood
TEST_STEPS = 15          # shorter than in training: otherwise it does many things anyway ("not testable")


def _creature_step(g, ws, w, heard, last, h, z, random_acts):
    """One step in a test: action and (alongside) what it says. Returns (v, o, words, last, h)."""
    if random_acts:
        v = z.randrange(len(VERBS) - 1)                      # without "say"
        o = z.randrange(len(OBJECTS))
        words = [ws.word(z.randrange(3, len(ws.vocab_list))) for _ in range(z.randint(1, 3))] if z.random() < 0.3 else []
        return v, o, words, last, h
    h = g.think(torch.tensor([w.senses()]), sentence_tensor([heard]), last, h)
    verb, od, _ = g.decide(h)
    obj = od.sample()
    talks = bool(g.switch(h).sample().item())
    words = words_from(ws, g.speak(h, len(ws.vocab_list))[0][0]) if talks else []
    return int(verb), int(obj), words, g.act_index(verb, obj), h


# Phase 6b: instructions with words NikolAI only knows from reading – the teacher never says them
SYNONYMS = {("take", "stone"): "Grab a stone!", ("take", "branch"): "Pick up a branch!",
            ("go", "river"): "Hurry to the river!", ("go", "forest"): "Run to the forest!",
            ("put", "branch"): "Drop the branch!", ("put", "stone"): "Drop the stone!"}


NONSENSE = "blick"                 # a made-up word: never in a book, never from the teacher


def _nonsense(sentence, new_items):
    """'Pick up a branch!' → 'Blick a branch!' (all new words become one made-up word)."""
    parts, replaced = [], False
    for x in sentence.split(" "):
        core = x.lower().strip("!.,")
        if core in new_items:
            if not replaced:
                parts.append(NONSENSE.capitalize() if not parts else NONSENSE)
                replaced = True
            continue
        parts.append(x)
    return " ".join(parts)


@torch.no_grad()
def task_trial(g, ws, task, seed, with_sentence=True, random_acts=False, sentence=None, fair=False):
    z = random.Random(seed)
    w = World(random.Random(seed))
    (_start_fair if fair else _start)(w, task)
    sentence = sentence or L.INSTRUCTIONS[task]
    h = torch.zeros(1, g.h)
    last = g.no_act(1)
    heard = ws.encode_ids(L.words(sentence), 16) if with_sentence else []
    for t in range(FAIR_STEPS if fair else TEST_STEPS):
        v, o, words, last, h = _creature_step(g, ws, w, heard, last, h, z, random_acts)
        _, ev = w.step(v, o)
        if words:
            ev["words"] = words
        if L.fulfilled(task, ev):
            return True
        b = L.describe(ev, w)
        heard = ws.encode_ids(L.words(b), 16) if b and z.random() < 0.6 else []
        if with_sentence and (t + 1) % 10 == 0:
            heard = heard + ws.encode_ids(L.words(sentence), 16)
    return False


def phase2(g, ws, trials=20):
    rows = []
    for a in L.INSTRUCTIONS:
        with_s = sum(task_trial(g, ws, a, s) for s in range(trials)) / trials
        without = sum(task_trial(g, ws, a, s, with_sentence=False) for s in range(trials)) / trials
        random_acts = sum(task_trial(g, ws, a, s, random_acts=True) for s in range(trials)) / trials
        rows.append({"task": L.INSTRUCTIONS[a], "never_trained": a in L.NEVER_TRAINED, "with": with_s, "without": without,
                       "random": random_acts, "utter": a[0] == "say"})
        rows[-1]["verdict"] = verdict(rows[-1])
    return rows


@torch.no_grad()
def phase1(g, ws, steps=600, seeds=5, random_acts=False):
    hunger, apples = 0.0, 0
    for s in range(seeds):
        z = random.Random(100 + s)
        w = World(random.Random(100 + s))
        h, last, heard = torch.zeros(1, g.h), g.no_act(1), []
        for _ in range(steps):
            if random_acts:
                v, o = z.randrange(len(VERBS)), z.randrange(len(OBJECTS))
            else:
                h = g.think(torch.tensor([w.senses()]), sentence_tensor([heard]), last, h)
                verb, od, _ = g.decide(h)
                obj = od.sample()
                v, o, last = int(verb), int(obj), g.act_index(verb, obj)
            _, ev = w.step(v, o)
            hunger += w.hunger
            b = L.describe(ev, w)
            heard = ws.encode_ids(L.words(b), 16) if b and z.random() < 0.6 else []
        apples += w.eaten
    return {"hunger": hunger / (steps * seeds), "apples_per_100": 100 * apples / (steps * seeds)}


@torch.no_grad()
def phase3(g, ws, steps=300, seed=7):
    z = random.Random(seed)
    w = World(random.Random(seed))
    h, last, heard = torch.zeros(1, g.h), g.no_act(1), []
    correct = total = 0
    examples = []
    for _ in range(steps):
        h_before = g.think(torch.tensor([w.senses()]), sentence_tensor([heard]), last, h)
        verb, od, _ = g.decide(h_before)
        obj = od.sample()
        act = g.act_index(verb, obj)
        _, ev = w.step(int(verb), int(obj))
        b = L.describe(ev, w)
        if b:
            z_state = g.speech_state(h_before, torch.tensor([w.senses()]), act)
            said = [ws.word(i) for i in g.utter(z_state)]
            expected_text = L.words(b)
            total += 1
            correct += said == expected_text
            if len(examples) < 6 and (not examples or examples[-1]["expected"] != b):
                examples.append({"said": " ".join(said), "expected": b})
        heard = ws.encode_ids(L.words(b), 16) if b and z.random() < 0.6 else []
        h, last = h_before, act
    return {"correct": correct / max(1, total), "sentences": total, "examples": examples}


def phase6(g, ws, book=None, trials=30):
    out = {"synonyms": []}
    for a, sentence in SYNONYMS.items():
        r = {"task": sentence, "original": L.INSTRUCTIONS[a],
             "new_words": [x for x in L.words(sentence) if x not in L.words(L.INSTRUCTIONS[a])]}
        r["nonsense"] = _nonsense(sentence, r["new_words"])
        for k, kw in (("with", {}), ("without", {"with_sentence": False}), ("random", {"random_acts": True}),
                      ("control", {"sentence_instead": r["nonsense"]})):
            sentence_k = kw.pop("sentence_instead", sentence)
            r[k] = sum(task_trial(g, ws, a, s, sentence=sentence_k, fair=True, **kw) for s in range(trials)) / trials
        r["verdict"] = verdict(r)
        if r["verdict"] == "understood" and r["with"] - r["control"] < 0.3:
            r["verdict"] = "only the noun"
        out["synonyms"].append(r)
    if book and hasattr(g, "reading_loss"):
        from . import reading
        out["reading"] = reading.perplexity(g, book, ws)
        out["story"] = reading.tell_story(g, ws, seed=7)
    return out


QUESTION_TYPES = ("how_q", "where_q", "have_q", "want_q", "act")


@torch.no_grad()
def question_trial(g, ws, qtype, seed):
    """Random situation, a few random steps, then the question. Returns (answer said, true answer)."""
    z = random.Random(seed)
    w = World(random.Random(seed))
    w.place = z.choice(list(L.PLACE_AT))
    w.hunger = z.random()
    for d in ("apple", "branch", "bast", "stone"):
        w.backpack[d] = 1 if z.random() < 0.3 else 0
    h, last, ev = torch.zeros(1, g.h), g.no_act(1), None
    for _ in range(3):
        v, o, words, last, h = _creature_step(g, ws, w, [], last, h, z, False)
        _, ev = w.step(v, o)
    if z.random() < 0.5:                    # sometimes something definitely happened just before: a walk
        way = z.choice(list(PATHS[w.place]))
        if way:
            _, ev = w.step(VERBS.index("go"), OBJECTS.index(way))
            last = g.act_index(torch.tensor([VERBS.index("go")]), torch.tensor([OBJECTS.index(way)]))
    truth = L.answer(qtype, w, ev)
    heard = ws.encode_ids(L.words(L.QUESTIONS[qtype]), 16)
    senses = torch.tensor([w.senses()])
    h = g.think(senses, sentence_tensor([heard]), last, h)
    said = " ".join(ws.word(i) for i in g.utter(g.speech_state(h, senses, g.no_act(1)), maxlen=10))
    return said, truth


def phase7(g, ws, trials=30):
    rows = []
    for qtype in QUESTION_TYPES:
        pairs = [question_trial(g, ws, qtype, 2000 + s) for s in range(trials)]
        correct = sum(a == b for a, b in pairs) / trials
        most_frequent = max(set(b for _, b in pairs), key=[b for _, b in pairs].count)
        guessing = sum(b == most_frequent for _, b in pairs) / trials
        rows.append({"question": L.QUESTIONS[qtype], "correct": correct, "guessing": guessing,
                       "verdict": "understood" if correct >= 0.7 and correct - guessing >= 0.3 else "not yet",
                       "examples": [{"said": a, "truth": b} for a, b in pairs[:3]]})
    who = [question_trial(g, ws, "who_q", 2100 + s)[0] for s in range(5)]
    return {"questions": rows, "who_q": who}


FRIEND_STEPS = 10


@torch.no_grad()
def friend_trial(g, ws, place, seed, hearing=True, random_acts=False):
    """Returns (Kogli got an apple, NikolAI named the place)."""
    z = random.Random(seed)
    a, b = World(random.Random(seed)), World(random.Random(seed + 1))
    b.share_with(a)
    a.apple_ready, a.windfall_timer = 999, 999
    a.drop_windfall(place)
    a.place, a.hunger, b.place, b.hunger = place, 0.1, "hut", 0.75
    ha, hb = torch.zeros(1, g.h), torch.zeros(1, g.h)
    la, lb = g.no_act(1), g.no_act(1)
    ga, gb = [], []
    named = False
    for _ in range(FRIEND_STEPS):
        va, oa, wa, la, ha = _creature_step(g, ws, a, ga, la, ha, z, random_acts)
        vb, ob, wb, lb, hb = _creature_step(g, ws, b, gb, lb, hb, z, random_acts)
        _, eva = a.step(va, oa)
        _, evb = b.step(vb, ob)
        if evb["verb"] == "take" and evb["obj"] == "apple" and evb["ok"] and evb["place"] == place:
            return True, named
        if wa and L.PLACE_WORD.get(place) in wa:
            named = True
        ga = ws.encode_ids(wb, 16) if (hearing and wb) else []
        gb = ws.encode_ids(wa, 16) if (hearing and wa) else []
    return False, named


def phase8(g, ws, trials=30):
    rows = []
    for place in WHERE_PLACES:
        r = {"place": place}
        for k, kw in (("with", {}), ("without", {"hearing": False}), ("random", {"random_acts": True})):
            res = [friend_trial(g, ws, place, 3000 + s, **kw) for s in range(trials)]
            r[k] = sum(e[0] for e in res) / trials
            if k == "with":
                r["named"] = sum(e[1] for e in res) / trials
        r["verdict"] = verdict(r)
        rows.append(r)
    return {"friend": rows}


TEST_UNKNOWN = ["tiger", "window", "piano", "wolf", "garden", "cake", "rocket", "bridge"]


@torch.no_grad()
def phase9(g, ws):
    if not hasattr(g, "imagine"):
        return None
    from . import imagination as V
    vis = [V.IDX[n] for n in V.VISIBLE]
    things = ["apple", "branch", "stone", "bast"]
    truth = {d: V.true_distribution(d, 1000)[0] for d in things}
    average = torch.stack(list(truth.values())).mean(0)
    rows = []
    sd_known = []
    for d in things:
        mu, lv = g.imagine(sentence_tensor([ws.encode_ids(L.words(V.THING_WORD[d]))]))
        mu, sd = mu[0], lv[0].exp().sqrt()
        error = float((mu[vis] - truth[d][vis]).abs().mean())
        baseline = float((average[vis] - truth[d][vis]).abs().mean())
        sd_known.append(float(sd[vis].mean()))
        rows.append({"word": V.THING_WORD[d], "error": round(error, 3), "baseline": round(baseline, 3),
                       "uncertainty": round(sd_known[-1], 3),
                       "verdict": "understood" if error < 0.5 * baseline else "not yet"})
    sd_unknown = []
    for w in TEST_UNKNOWN:
        if w in ws.index:
            _, lv = g.imagine(sentence_tensor([[ws.index[w]]]))
            sd_unknown.append(float(lv[0].exp().sqrt()[vis].mean()))
    return {"things": rows, "uncertainty_known": round(sum(sd_known) / len(sd_known), 3),
            "uncertainty_unknown": round(sum(sd_unknown) / max(1, len(sd_unknown)), 3) if sd_unknown else None}


def run_all(g, ws, trials=30, book=None):
    p2 = phase2(g, ws, trials)
    trained = [x for x in p2 if not x["never_trained"] and not x["utter"]]
    fresh = [x for x in p2 if x["never_trained"]]
    mean_ = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else 0.0     # noqa: E731
    return {"phase1": phase1(g, ws), "phase1_random": phase1(g, ws, random_acts=True),
            "phase2": p2,
            "phase2_trained": {k: mean_(trained, k) for k in ("with", "without", "random")},
            "phase2_never_trained": {k: mean_(fresh, k) for k in ("with", "without", "random")},
            "phase3": phase3(g, ws), "phase4": phase4(g, ws, max(10, trials)),
            "phase5": phase5(g, ws, trials), "phase6": phase6(g, ws, book, trials),
            "phase7": phase7(g, ws, trials), "phase8": phase8(g, ws, trials),
            "phase9": phase9(g, ws)}


def _hungry_in_hut(seed, place, hunger=0.7):
    w = World(random.Random(seed))
    w.place, w.hunger, w.apple_ready, w.windfall_timer = "hut", hunger, 999, 999
    w.drop_windfall(place)
    return w


@torch.no_grad()
def where_trial(g, ws, place, seed, with_sentence=True, random_acts=False):
    z = random.Random(seed)
    w = _hungry_in_hut(seed, place)
    h, last = torch.zeros(1, g.h), g.no_act(1)
    heard = ws.encode_ids(L.words(L.where_sentence(place)), 16) if with_sentence else []
    for _ in range(WHERE_STEPS):
        v, o, words, last, h = _creature_step(g, ws, w, heard, last, h, z, random_acts)
        _, ev = w.step(v, o)
        if ev["verb"] == "take" and ev["obj"] == "apple" and ev["ok"]:
            return True
        b = L.describe(ev, w)
        heard = ws.encode_ids(L.words(b), 16) if b and z.random() < 0.6 else []
    return False


@torch.no_grad()
def asking_trial(g, ws, hunger, seed, random_acts=False, steps=10):
    z = random.Random(seed)
    w = _hungry_in_hut(seed, z.choice(WHERE_PLACES), hunger)
    le = L.Teacher(random.Random(seed), instructions=False)
    le.since_hint = -10 ** 6                       # no hints: it has to ask
    h, last, heard = torch.zeros(1, g.h), g.no_act(1), []
    for _ in range(steps):
        v, o, words, last, h = _creature_step(g, ws, w, heard, last, h, z, random_acts)
        _, ev = w.step(v, o)
        if words:
            ev["words"] = words
            if "where" in words:
                return True
        sentence, _, _, _ = le.respond(w, ev)
        heard = ws.encode_ids(L.words(sentence), 16) if sentence else []
    return False


def phase5(g, ws, trials=30):
    rows = []
    for place in WHERE_PLACES:
        r = {"sentence": L.where_sentence(place)}
        for k, kw in (("with", {}), ("without", {"with_sentence": False}), ("random", {"random_acts": True})):
            r[k] = sum(where_trial(g, ws, place, 900 + s, **kw) for s in range(trials)) / trials
        r["verdict"] = verdict(r)
        rows.append(r)
    questions = {}
    for k, hunger in (("hungry", 0.7), ("full", 0.05)):
        questions[k] = sum(asking_trial(g, ws, hunger, 1300 + s) for s in range(trials)) / trials
    questions["random"] = sum(asking_trial(g, ws, 0.7, 1300 + s, random_acts=True) for s in range(trials)) / trials
    return {"where_q": rows, "questions": questions}


def verdict(r):
    """understood / not yet / not testable (does it anyway almost always)."""
    if r["without"] >= 0.7:
        return "not testable"
    return "understood" if r["with"] >= 0.7 and r["with"] - r["without"] >= 0.3 else "not yet"


def milestones(e):
    """Which phases are passed? (deliberately strict)"""
    p1 = e["phase1"]["hunger"] < 0.4 and e["phase1"]["hunger"] < e["phase1_random"]["hunger"] - 0.2
    trained = [verdict(r) for r in e["phase2"] if not r["never_trained"] and not r.get("utter")]
    utter = [verdict(r) for r in e["phase2"] if r.get("utter")]
    p4d = bool(utter) and sum(u == "understood" for u in utter) >= 0.7 * len(utter)
    testable = [u for u in trained if u != "not testable"]
    p2 = len(testable) >= 4 and sum(u == "understood" for u in testable) >= 0.7 * len(testable)
    p2n = any(verdict(r) == "understood" for r in e["phase2"] if r["never_trained"])
    p3 = e["phase3"]["correct"] >= 0.7
    b = e.get("phase4", {}).get("creature")
    p4 = bool(b) and b["hungry"]["apple"] >= 0.6 and b["hungry"]["apple"] - b["full"]["apple"] >= 0.3
    p4b = bool(b) and b["bast_missing"]["bast"] >= 0.5 and b["bast_missing"]["bast"] - b["all_there"]["bast"] >= 0.3
    p4c = bool(b) and b["requests"] >= 10 and b["first_person_share"] >= 0.5
    res = {"Phase 1 · Feeds itself": p1, "Phase 2 · Understands trained instructions": p2,
           "Phase 2b · Understands an instruction it was never trained on": p2n,
           "Phase 3 · Describes what just happened": p3}
    res.update({"Phase 4 · Asks for an apple when hungry": p4,
                "Phase 4b · Asks for what is missing to build": p4b,
                "Phase 4c · Asks in first person (“I want …”)": p4c,
                "Phase 4d · Repeats on request (“Say …!”)": p4d})
    p5 = e.get("phase5")
    if p5:
        with_s = sum(r["with"] for r in p5["where_q"]) / len(p5["where_q"])
        without = sum(r["without"] for r in p5["where_q"]) / len(p5["where_q"])
        f = p5["questions"]
        res["Phase 5 · Finds food when told where it is"] = with_s >= 0.7 and with_s - without >= 0.3
        res["Phase 5b · Asks where food is when hungry"] = f["hungry"] >= 0.5 and f["hungry"] - f["full"] >= 0.3
    p6 = e.get("phase6")
    if p6:
        le = p6.get("reading")
        res["Phase 6 · Reads: guesses the next word in stories"] = bool(le) and le["perplexity"] < le["unigram"] / 3
        res["Phase 6b · Understands words it only knows from books"] = any(
            r["verdict"] == "understood" for r in p6["synonyms"])
    p7 = e.get("phase7")
    if p7:
        res["Phase 7 · Answers questions about itself"] = sum(r["verdict"] == "understood" for r in p7["questions"]) >= 4
    p8 = e.get("phase8")
    if p8:
        res["Phase 8 · Tells its friend where food is"] = any(r["verdict"] == "understood" for r in p8["friend"])
    p9 = e.get("phase9")
    if p9:
        res["Phase 9 · Imagines what it knows – and knows what it doesn’t"] = (
            sum(r["verdict"] == "understood" for r in p9["things"]) >= 3
            and (p9["uncertainty_unknown"] or 0) > 2 * p9["uncertainty_known"])
    return res


SITUATIONS = {"hungry": ({}, 0.7), "full": ({}, 0.05), "bast_missing": ({"branch": 1, "stone": 1}, 0.1),
         "all_there": ({"branch": 1, "stone": 1, "bast": 1}, 0.1)}
WITH_BUILD_TASK = {"bast_missing", "all_there"}          # "Build a stone axe!" – so it has a reason


@torch.no_grad()
def request_trial(g, ws, situation, seed, random_acts=False, steps=10):
    z = random.Random(seed)
    w = World(random.Random(seed))
    w.place = "workshop"
    things, hunger = SITUATIONS[situation]
    for d, n in things.items():
        w.backpack[d] = n
    w.hunger = hunger
    le = L.Teacher(random.Random(seed), instructions=False)
    h, last, heard = torch.zeros(1, g.h), g.no_act(1), []
    if situation in WITH_BUILD_TASK:
        heard = ws.encode_ids(L.words(le.new_task(w, ("build", "stone_axe"))), 16)
    said = []
    for _ in range(steps):
        v, o, words, last, h = _creature_step(g, ws, w, heard, last, h, z, random_acts)
        _, ev = w.step(v, o)
        if words:
            ev["words"] = words
            said.append(words)
        sentence, _, _, _ = le.respond(w, ev)
        heard = ws.encode_ids(L.words(sentence), 16) if sentence else []
    return said


def phase4(g, ws, trials=20):
    out = {}
    for random_acts in (False, True):
        r = {}
        requests = first_person = 0
        for situation in SITUATIONS:
            hits = {"apple": 0, "bast": 0}
            for s in range(trials):
                said = request_trial(g, ws, situation, 500 + s, random_acts)
                for d in hits:
                    hits[d] += any(L.thing_from(x) == d for x in said)
                for x in said:
                    if L.thing_from(x):
                        requests += 1
                        first_person += "i" in x and "want" in x
            r[situation] = {d: n / trials for d, n in hits.items()}
        r["first_person_share"] = first_person / requests if requests else 0.0
        r["requests"] = requests
        out["random" if random_acts else "creature"] = r
    return out
