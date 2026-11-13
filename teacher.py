"""The teacher: talks to NikolAI about what it is really doing right now – like a parent with a toddler.

* Describes actions ("You take an apple.") and names things ("This is the apple tree.").
* Gives instructions ("Go to the forest!") and praises NikolAI when it follows them.
* Some combinations are never given as instructions during training (e.g. "Put down the apple!"). They only appear
  in tests: that shows whether NikolAI really understood the words – or only memorised whole sentences.
* Listens: if NikolAI says the word for something it really needs right now ("apple" when hungry, "bast" at the
  workbench when bast is the only thing missing for a stone axe), the teacher hands it over and models the full
  sentence in first person ("Say: I want an apple.").
* Does not give everything: apples only now and then (NikolAI still has to pick them itself), a stone axe never
  (it has to build one), branch, stone and bast only for an instruction, and nothing it doesn't need. Receiving
  something is never rewarded in itself – only what NikolAI achieves with it counts (otherwise it learns to trick
  the teacher, as happened in version 3).

Version 6:
* Knows where the windfall apple lies – NikolAI doesn't. When NikolAI is hungry, the teacher sometimes says where it
  is ("There is an apple at the river.") and sometimes models the question ("Hungry? Say: where is an apple?").
  If NikolAI asks "where … apple", the teacher answers. Listening (or asking) now really pays off.
* Points at the windfall apple when NikolAI arrives where it lies ("Look, there is an apple here. Take an apple!"):
  at first NikolAI only knows apples from the tree. Hindsight then links this to "There is an apple in the forest."
* Gives instructions against habit: what NikolAI does on its own anyway is asked for less often, and nothing about
  apples while it is hungry anyway. Otherwise it can earn praise without listening.
* hindsight(): after the fact, what instruction would this action have fulfilled? ("You took a stone" → "Take a
  stone!"). Practising uses this to learn from every action, not only from the rare real instructions
  (hindsight relabeling: Andrychowicz et al. 2017; for language: Chan et al. 2019).

Version 8 – conversation: now and then the teacher asks NikolAI about itself ("Where are you?", "How are you?",
"What do you have?", "What do you want?", "What did you do?", "Who are you?"). antwort() is the true answer in its
world right now – NikolAI learns to give it (with the speaking part it got from reading).

Version 10 – a friend: NikolAI and Kogli share a world and hear what the other says. When the friend is hungry
and this one stands right where an apple lies, the teacher whispers: "Kogli is hungry. Say: there is an apple in the
forest." – the same sentence the teacher uses as a hint, which the friend already understands. Whether passing it on
is worth it, they find out themselves: each also feels it when the other gets full (a shared reward).

Everything the teacher says is correct English and true in the world. It knows nothing about the brain's insides.
"""

import re

from .world import RECIPE

PLACE_TO = {"hut": "to the hut", "orchard": "to the orchard", "forest": "to the forest", "river": "to the river",
            "workshop": "to the workshop"}
A = {"apple": "an apple", "branch": "a branch", "bast": "some bast", "stone": "a stone", "stone_axe": "a stone axe"}
THE = {"apple": "the apple", "branch": "the branch", "bast": "the bast", "stone": "the stone", "stone_axe": "the stone axe"}
NO = {"apple": "no apple", "branch": "no branch", "bast": "no bast", "stone": "no stone", "stone_axe": "no stone axe"}
STATION_NAME = {"orchard": "This is the apple tree.", "forest": "This is a tree.", "river": "This is the riverbank.",
                "workshop": "This is the workbench.", "hut": "This is your bed."}
WORD = {"apple": "apple", "branch": "branch", "bast": "bast", "stone": "stone", "stone_axe": "stone axe"}
WORD_TO_THING = {"apple": "apple", "branch": "branch", "bast": "bast", "stone": "stone", "axe": "stone_axe"}
DIRECTIONS = ("north", "south", "east", "west")
PLACE_AT = {"hut": "in the hut", "orchard": "in the orchard", "forest": "in the forest", "river": "at the river",
           "workshop": "in the workshop"}
QUESTIONS = {"how_q": "How are you?", "where_q": "Where are you?", "have_q": "What do you have?", "want_q": "What do you want?",
          "act": "What did you do?", "who_q": "Who are you?"}
QUESTION_RATE = 0.10         # chance per step to ask something when no instruction is running
PLACE_WORD = {"hut": "hut", "orchard": "orchard", "forest": "forest", "river": "river", "workshop": "workshop"}
HINT_PAUSE = 20           # steps between two hints / modelled questions

# Instructions: (verb, object) → sentence. For "gehe" the object is the destination; for "sage" the words to say.
INSTRUCTIONS = {("take", d): f"Take {A[d]}!" for d in ("apple", "branch", "bast", "stone")}
INSTRUCTIONS.update({("go", o): f"Go {PLACE_TO[o]}!" for o in PLACE_TO})
INSTRUCTIONS.update({("put", d): f"Put down {THE[d]}!" for d in ("apple", "branch", "bast", "stone")})
INSTRUCTIONS[("eat", "apple")] = "Eat an apple!"
INSTRUCTIONS[("build", "stone_axe")] = "Build a stone axe!"
INSTRUCTIONS.update({("say", WORD[d]): f"Say “{WORD[d]}”!" for d in WORD})                       # repeating
INSTRUCTIONS.update({("say", f"i want {A[d]}"): f"Say “I want {A[d]}”!" for d in ("apple", "stone", "bast")})
APPLE_PAUSE = 150         # steps between two apples from the teacher (rare: begging alone must not feed it)
GIVE_PAUSE = 8
NEVER_TRAINED = {("put", "apple"), ("take", "bast")}      # test only – NikolAI knows the words from other sentences
DEADLINE = 30                                              # steps NikolAI has to follow an instruction


def words(sentence):
    return re.findall(r"[a-z]+", (sentence or "").lower())


def prompted_words(sentence):
    """What the teacher asks NikolAI to repeat ("Say: I want an apple." / "Say “stone”!") – or None."""
    m = re.search(r"Say:? “?([^“”!.]+)", sentence or "")
    return words(m.group(1)) if m else None


def describe(ev, world):
    """What the teacher says about an action (or None)."""
    v, o, ok = ev["verb"], ev["obj"], ev["ok"]
    if v in ("wait", "say"):
        return None
    if v == "go":
        if o not in DIRECTIONS:
            return "You can walk north, south, east or west."
        return f"You go {PLACE_TO[ev['target']]}." if ok else "There is no path that way."
    if o in DIRECTIONS:
        return "That doesn't work."
    if v == "take":
        if ok:
            return f"You take {A[o]}."
        return f"There is {NO[o]} here." if world.backpack[o] < 3 else "You can't carry any more."
    if v == "eat":
        if o != "apple":
            return "You can't eat that."
        return "You eat an apple. Yummy!" if ok else "You have no apple."
    if v == "put":
        return f"You put down {THE[o]}." if ok else f"You have {NO[o]}."
    if v == "build":
        if o not in RECIPE:
            return "That can't be built."
        if ok:
            return "You build a stone axe!"
        if ev["place"] != "workshop":
            return "You can build things at the workbench."
        return "For that you need a branch, a stone and some bast."
    return None


def answer(qtype, world, ev=None, name="nikolai"):
    """The true answer to a question about itself, right now, in first person."""
    r = world.backpack
    if qtype == "how_q":
        return "i am hungry" if world.hunger > 0.6 else "i am full" if world.hunger < 0.25 else "i am fine"
    if qtype == "where_q":
        return f"i am {PLACE_AT[world.place]}"
    if qtype == "have_q":
        things = [A[d] for d in ("apple", "branch", "bast", "stone", "stone_axe") if r[d] > 0]
        return "i have " + " and ".join(things[:2]) if things else "i have nothing"
    if qtype == "want_q":
        if world.hunger > 0.5 and r["apple"] == 0:
            return "i want an apple"
        missing = [d for d in ("branch", "stone", "bast") if r[d] == 0]
        if r["stone_axe"] == 0 and 0 < len(missing) < 3:
            return f"i want {A[missing[0]]}"
        return "i want to play"
    if qtype == "act":
        if not ev or not ev["ok"] or ev["verb"] in ("wait", "say"):
            return "i did nothing"
        v, o = ev["verb"], ev["obj"]
        return {"go": lambda: f"i went {PLACE_TO.get(ev.get('target'), '')}", "take": lambda: f"i took {A[o]}",
                "eat": lambda: "i ate an apple", "put": lambda: f"i put down {THE[o]}",
                "build": lambda: "i built a stone axe"}[v]()
    return f"i am {name}"


def question_from_text(text):
    """Which of the teacher's questions is this? (also when a human asks it)"""
    t = _norm(text)
    for qtype, sentence in QUESTIONS.items():
        if t == _norm(sentence):
            return qtype
    return None


def where_sentence(place):
    return f"There is an apple {PLACE_AT[place]}."


def action_of(ev):
    """Which instruction (verb, obj) does this successful action fulfil? Or None."""
    if not ev or not ev["ok"]:
        return None
    a = ("go", ev["target"]) if ev["verb"] == "go" and "target" in ev else (ev["verb"], ev["obj"])
    return a if a in INSTRUCTIONS else None


def hindsight(ev):
    """Sentences that would have been true instructions or hints for what NikolAI just did (never test-only ones)."""
    out = []
    a = action_of(ev)
    if a and a not in NEVER_TRAINED:
        out.append(INSTRUCTIONS[a])
    if ev and ev.get("windfall"):
        out.append(where_sentence(ev["place"]))
    return out


def fulfilled(task, ev):
    v, o = task
    if v == "say":
        return bool(ev.get("words")) and all(x in ev["words"] for x in o.split())
    if v == "go":
        return ev["place_after"] == o and ev["verb"] == "go" and ev["ok"]
    return ev["verb"] == v and ev["obj"] == o and ev["ok"]


def thing_from(words):
    """Which thing does an utterance ask for? ("stone axe" counts as the axe, not the stone.)"""
    if "axe" in words:
        return "stone_axe"
    return next((WORD_TO_THING[w] for w in words if w in WORD_TO_THING), None)


class Teacher:
    """A teacher for one world. sprich() returns what NikolAI hears this step, and the reward."""

    def __init__(self, rng, instructions=True, trained_only=True, describe_rate=0.6, name="nikolai"):
        self.z = rng
        self.name = name
        self.friend = None                                   # (world of the friend, its name) – version 10
        self.friend_said = []                               # what the friend just said (NikolAI heard it too)
        self.instructions_on = instructions
        self.trained_only = trained_only
        self.rate = describe_rate
        self.task = None
        self.since = 0
        self.pause = 0
        self.successes = 0
        self.trials = 0
        self.since_give = 99
        self.since_apple = 99
        self.requests = {"recognized": 0, "first_person": 0, "given": 0, "where_q": 0}
        self.since_hint = 99
        self.question = None                                    # the question just asked (NikolAI answers next step)
        self.habits = {}                                    # what NikolAI does on its own (no instruction)
        self._b0 = 0
        self._w0 = 0

    def new_task(self, world, task=None):
        if task is None:
            possible = [a for a in INSTRUCTIONS if not (self.trained_only and a in NEVER_TRAINED)]
            possible = [a for a in possible if not (a[0] == "go" and a[1] == world.place)]
            possible = [a for a in possible if a[0] != "put" or world.backpack[a[1]] > 0] or possible
            if world.hunger > 0.5:                            # it would fetch and eat apples anyway
                possible = [a for a in possible if a[1] != "apple"] or possible
            weights = [1.0 / (1.0 + self.habits.get(a, 0)) for a in possible]
            task = self.z.choices(possible, weights)[0]
            self.habits = {k: v / 2 for k, v in self.habits.items()}
        self.task, self.since = task, 0
        self.trials += 1
        return INSTRUCTIONS[task]

    # ------------------------------------------------------------ listening
    def needs(self, world, thing):
        """Does NikolAI really need this thing right now? (Only then does the teacher hand it over.)"""
        r = world.backpack
        if thing == "apple":
            return world.hunger > 0.4 and r["apple"] == 0 and self.since_apple >= APPLE_PAUSE
        if thing == "stone_axe":
            return False
        if r[thing] > 0:
            return False
        if self.task in (("take", thing), ("put", thing)):
            return True
        if self.task != ("build", "stone_axe"):         # only for a real purpose
            return False
        others = [d for d in ("branch", "stone", "bast") if d != thing]
        return all(r[d] > 0 for d in others)              # the only thing still missing for a stone axe

    def listen_to_creature(self, world, words):
        """NikolAI said something. Returns (answer, reward)."""
        if "where" in words:                               # a question, not a request
            self.requests["where_q"] += 1
            self.since_hint = 0
            if world.windfall:
                return where_sentence(world.windfall), 0.0
            return "No apple is lying around. Look at the apple tree.", 0.0
        x = thing_from(words)
        if not x:
            return ("Hm?" if words and self.z.random() < 0.3 else ""), 0.0
        first_person = "i" in words and "want" in words
        self.requests["recognized"] += 1
        if x == "stone_axe" and world.backpack["stone_axe"] == 0:
            return "You have to build a stone axe yourself: from a branch, a stone and some bast.", 0.0
        if self.since_give < GIVE_PAUSE or not self.needs(world, x):
            if world.backpack[x] > 0:
                return f"You already have {A[x]}.", 0.0
            return f"{WORD[x].capitalize()}? Not now.", 0.0
        world.give(x)
        self.since_give = 0
        if x == "apple":
            self.since_apple = 0
        self.requests["given"] += 1
        if first_person:                                            # no reward for receiving – only what it achieves counts
            self.requests["first_person"] += 1
            return f"Nicely said! Here is {A[x]}.", 0.0
        return f"You want {A[x]}? Here is {A[x]}. Say: I want {A[x]}.", 0.0

    def respond(self, world, ev):
        """(sentence heard or "", reward from the teacher, description or None, instruction done?)"""
        parts, reward, done = [], 0.0, False
        self.question = None
        self.since_give += 1
        self.since_apple += 1
        self.since_hint += 1
        if ev and ev.get("words"):
            answer, reward = self.listen_to_creature(world, ev["words"])
            if answer:
                parts.append(answer)
        if ev and ev["verb"] == "go" and ev["ok"] and ev.get("target") == world.windfall and not self.task \
                and world.backpack["apple"] < 2:
            parts.append("Look, there is an apple here.")       # scaffolding: points at what it can't connect yet
            parts.append(self.new_task(world, ("take", "apple")))
        if not self.task:
            a = action_of(ev)
            if a:
                self.habits[a] = self.habits.get(a, 0) + 1
        if self.friend and self.since_hint >= HINT_PAUSE and world.windfall == world.place:
            fw, fname = self.friend                          # it sees an apple – and its friend needs one
            if fw.backpack["apple"] == 0 and fw.place != world.place:
                if "where" in self.friend_said:             # the friend asked – NikolAI heard that itself
                    self.since_hint = 0
                    parts.append(f"{fname.capitalize()} asks. Say: {where_sentence(world.place).lower()}")
                elif fw.hunger > 0.45:
                    self.since_hint = 0
                    parts.append(f"{fname.capitalize()} is hungry. Say: {where_sentence(world.place).lower()}")
        if world.windfall and world.hunger > 0.35 and world.backpack["apple"] == 0 and self.since_hint >= HINT_PAUSE \
                and world.place != world.windfall:
            self.since_hint = 0
            direct = 0.2 if self.friend else 0.5             # with a friend around, the friend should tell
            parts.append(where_sentence(world.windfall) if self.z.random() < direct else "Hungry? Say: where is an apple?")
        descr = describe(ev, world) if ev else None
        name = STATION_NAME.get(ev["target"]) if ev and ev["verb"] == "go" and ev["ok"] and self.z.random() < 0.3 else None
        if descr and self.z.random() < self.rate:
            parts.append(descr)
        else:
            descr = None
        if name:
            parts.append(name)
        if self.task:
            self.since += 1
            if ev and fulfilled(self.task, ev):
                parts.append("Well done!")
                reward, done = reward + 1.0, True
                self.successes += 1
                self.task, self.pause = None, self.z.randint(5, 25)
            elif self.since >= DEADLINE:
                self.task, self.pause = None, self.z.randint(5, 25)
            elif self.since % 10 == 0:
                parts.append(INSTRUCTIONS[self.task])            # say it again
        elif self.instructions_on:
            self.pause -= 1
            if self.pause <= 0:
                parts.append(self.new_task(world))
            elif self.z.random() < QUESTION_RATE:            # a little conversation
                # (they share one brain – it cannot tell which body it is, so only NikolAI is asked its name)
                self.question = self.z.choice([f for f in QUESTIONS if f != "who_q" or self.name == "nikolai"])
                parts.append(QUESTIONS[self.question])
        return " ".join(parts), reward, descr, done


def all_sentences():
    """Everything the teacher can say (for the reading vocabulary: its words must all be known)."""
    out = list(INSTRUCTIONS.values()) + list(STATION_NAME.values()) + [where_sentence(o) for o in PLACE_AT]
    out += ["You can walk north, south, east or west.", "There is no path that way.", "That doesn't work.",
            "You can't carry any more.", "You can't eat that.", "You eat an apple. Yummy!", "You have no apple.",
            "That can't be built.", "You build a stone axe!", "You can build things at the workbench.",
            "For that you need a branch, a stone and some bast.", "Well done!", "Hm?", "Nicely said!",
            "Hungry? Say: where is an apple?", "Look, there is an apple here.",
            "No apple is lying around. Look at the apple tree.",
            "You have to build a stone axe yourself: from a branch, a stone and some bast."]
    out += list(QUESTIONS.values()) + ["i am nikolai", "i am kogli", "i did nothing", "i want to play", "i have nothing",
                                     "Nikolai is hungry.", "Kogli is hungry.", "Nikolai asks.", "Kogli asks."]
    for d in A:
        out += [f"You take {A[d]}.", f"There is {NO[d]} here.", f"You put down {THE[d]}.", f"You have {NO[d]}.",
                f"You already have {A[d]}.", f"{WORD[d].capitalize()}? Not now.", f"Here is {A[d]}.",
                f"You want {A[d]}? Here is {A[d]}. Say: I want {A[d]}."]
    out += [f"You go {z}." for z in PLACE_TO.values()]
    return out


def _norm(t):
    return " ".join(words(t))


def task_from_text(text):
    """Is this one of the teacher's instructions? Then the human becomes the teacher. Returns (verb, obj) or None."""
    t = _norm(text)
    for a, sentence in INSTRUCTIONS.items():
        if t == _norm(sentence):
            return a
    return None
